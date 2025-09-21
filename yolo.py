# Custom YOLO Model Training System - Fixed Version
# Focus: Smart Dataset Processing & Custom Model Training with Device Detection
# Requirements: pip install streamlit ultralytics opencv-python pillow numpy pycocotools torch

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import yaml
from pathlib import Path
import zipfile
import shutil
import json
import glob
from typing import List, Dict, Tuple
import random
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import threading
import time
import torch


def get_available_devices():
    """Get list of available devices for training"""
    devices = ['cpu']

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            try:
                gpu_name = torch.cuda.get_device_name(i)
                devices.append(f"cuda:{i} ({gpu_name})")
            except:
                devices.append(f"cuda:{i}")

        if device_count > 1:
            devices.append(f"multi-gpu (0-{device_count - 1})")

    return devices


def format_device_selection(selected_device):
    """Convert UI device selection to YOLO format"""
    if selected_device == 'cpu':
        return 'cpu'
    elif selected_device.startswith('cuda:'):
        return selected_device.split(' ')[0].replace('cuda:', '')
    elif selected_device.startswith('multi-gpu'):
        device_count = torch.cuda.device_count()
        return ','.join([str(i) for i in range(device_count)])
    else:
        return 'cpu'


class SmartDatasetProcessor:
    """Advanced dataset processor with format detection and conversion"""

    def __init__(self):
        self.supported_img_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.supported_label_formats = ['.txt', '.json', '.xml']
        self.dataset_stats = {}

    def analyze_dataset(self, dataset_path: str) -> Dict:
        """Comprehensive dataset analysis"""
        analysis = {
            'total_images': 0,
            'total_labels': 0,
            'format': 'unknown',
            'class_distribution': {},
            'image_sizes': [],
            'issues': []
        }

        try:
            # Find all images
            image_files = []
            for ext in self.supported_img_formats:
                image_files.extend(glob.glob(os.path.join(dataset_path, "**", f"*{ext}"), recursive=True))

            analysis['total_images'] = len(image_files)

            # Find all labels
            label_files = []
            for ext in self.supported_label_formats:
                label_files.extend(glob.glob(os.path.join(dataset_path, "**", f"*{ext}"), recursive=True))

            analysis['total_labels'] = len(label_files)

            # Detect format
            analysis['format'] = self.detect_dataset_format(dataset_path)

            # Analyze image sizes (sample first 50 images)
            sample_images = image_files[:min(50, len(image_files))]
            for img_path in sample_images:
                try:
                    with Image.open(img_path) as img:
                        analysis['image_sizes'].append(img.size)
                except Exception:
                    analysis['issues'].append(f"Could not read image: {os.path.basename(img_path)}")

            # Analyze class distribution for YOLO format
            if analysis['format'] == 'yolo_txt':
                class_counts = {}
                for img_path in image_files:
                    label_path = img_path.rsplit('.', 1)[0] + '.txt'
                    if os.path.exists(label_path):
                        try:
                            with open(label_path, 'r') as f:
                                for line in f:
                                    if line.strip():
                                        class_id = int(line.split()[0])
                                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        except Exception:
                            analysis['issues'].append(f"Error reading label: {os.path.basename(label_path)}")

                analysis['class_distribution'] = class_counts

        except Exception as e:
            analysis['issues'].append(f"Analysis error: {str(e)}")

        return analysis

    def detect_dataset_format(self, dataset_path: str) -> str:
        """Detect dataset format with better accuracy"""
        label_files = []
        for ext in self.supported_label_formats:
            label_files.extend(glob.glob(os.path.join(dataset_path, "**", f"*{ext}"), recursive=True))

        if not label_files:
            return "images_only"

        # Check multiple label files for better detection
        format_votes = {}

        for label_file in label_files[:5]:  # Check first 5 files
            if label_file.endswith('.txt'):
                try:
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                        if content:
                            lines = content.split('\n')
                            first_line = lines[0].split()
                            if len(first_line) >= 5 and all(self.is_float(x) for x in first_line[:5]):
                                if 0 <= float(first_line[1]) <= 1 and 0 <= float(first_line[2]) <= 1:
                                    format_votes['yolo_txt'] = format_votes.get('yolo_txt', 0) + 1
                                else:
                                    format_votes['absolute_txt'] = format_votes.get('absolute_txt', 0) + 1
                            else:
                                format_votes['custom_txt'] = format_votes.get('custom_txt', 0) + 1
                except Exception:
                    continue
            elif label_file.endswith('.json'):
                try:
                    with open(label_file, 'r') as f:
                        data = json.load(f)
                        if 'annotations' in data and 'images' in data:
                            format_votes['coco_json'] = format_votes.get('coco_json', 0) + 1
                        else:
                            format_votes['custom_json'] = format_votes.get('custom_json', 0) + 1
                except Exception:
                    continue

        # Return most voted format
        if format_votes:
            return max(format_votes.items(), key=lambda x: x[1])[0]

        return "unknown"

    def is_float(self, value: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def convert_to_yolo_format(self, dataset_path: str, output_path: str,
                               train_split: float = 0.8, val_split: float = 0.15) -> Tuple[str, List[str], Dict]:
        """Convert any dataset format to YOLO format with detailed statistics"""

        # Create directory structure
        dirs = {
            'train_img': os.path.join(output_path, 'images', 'train'),
            'val_img': os.path.join(output_path, 'images', 'val'),
            'test_img': os.path.join(output_path, 'images', 'test'),
            'train_label': os.path.join(output_path, 'labels', 'train'),
            'val_label': os.path.join(output_path, 'labels', 'val'),
            'test_label': os.path.join(output_path, 'labels', 'test')
        }

        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Find all images
        image_files = []
        for ext in self.supported_img_formats:
            image_files.extend(glob.glob(os.path.join(dataset_path, "**", f"*{ext}"), recursive=True))

        if not image_files:
            raise ValueError("No image files found in the dataset")

        dataset_format = self.detect_dataset_format(dataset_path)
        class_names = set()
        processed_files = []
        conversion_stats = {
            'total_processed': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'empty_labels': 0,
            'class_distribution': {}
        }

        # Process based on detected format
        if dataset_format == "yolo_txt":
            processed_files, class_names = self._process_yolo_format(image_files, conversion_stats)
        elif dataset_format in ["coco_json", "custom_json"]:
            processed_files, class_names = self._process_json_format(dataset_path, image_files, conversion_stats)
        elif dataset_format == "images_only":
            processed_files, class_names = self._process_images_only(image_files, conversion_stats)
        else:
            # Try to process as custom format
            processed_files, class_names = self._process_custom_format(dataset_path, image_files, conversion_stats)

        # Split dataset
        random.shuffle(processed_files)

        test_split = 1.0 - train_split - val_split
        train_end = int(len(processed_files) * train_split)
        val_end = int(len(processed_files) * (train_split + val_split))

        train_files = processed_files[:train_end]
        val_files = processed_files[train_end:val_end]
        test_files = processed_files[val_end:]

        # Copy files to respective directories
        self._copy_files_to_split(train_files, dirs['train_img'], dirs['train_label'])
        self._copy_files_to_split(val_files, dirs['val_img'], dirs['val_label'])
        if test_files:
            self._copy_files_to_split(test_files, dirs['test_img'], dirs['test_label'])

        # Create data.yaml
        class_names_list = sorted(list(class_names)) if class_names else ['object']
        yaml_config = {
            'path': output_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test' if test_files else None,
            'nc': len(class_names_list),
            'names': class_names_list
        }

        yaml_path = os.path.join(output_path, 'data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)

        # Update statistics
        conversion_stats.update({
            'train_count': len(train_files),
            'val_count': len(val_files),
            'test_count': len(test_files),
            'total_classes': len(class_names_list),
            'dataset_format': dataset_format
        })

        return yaml_path, class_names_list, conversion_stats

    def _process_yolo_format(self, image_files, stats):
        """Process YOLO format files"""
        processed_files = []
        class_names = set()

        for img_path in image_files:
            label_path = img_path.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            for line in content.split('\n'):
                                if line.strip():
                                    class_id = int(line.split()[0])
                                    class_names.add(f"class_{class_id}")
                                    stats['class_distribution'][class_id] = stats['class_distribution'].get(class_id,
                                                                                                            0) + 1
                        else:
                            stats['empty_labels'] += 1
                    processed_files.append((img_path, label_path))
                    stats['successful_conversions'] += 1
                except Exception:
                    stats['failed_conversions'] += 1
            else:
                # Create empty label file
                temp_label = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                temp_label.close()
                processed_files.append((img_path, temp_label.name))
                stats['empty_labels'] += 1

            stats['total_processed'] += 1

        return processed_files, class_names

    def _process_json_format(self, dataset_path, image_files, stats):
        """Process JSON format files (COCO or custom)"""
        processed_files = []
        class_names = set()

        json_files = glob.glob(os.path.join(dataset_path, "**", "*.json"), recursive=True)

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                # Process COCO format
                if 'annotations' in data and 'images' in data and 'categories' in data:
                    processed, classes = self._process_coco_json(data, image_files, stats)
                    processed_files.extend(processed)
                    class_names.update(classes)

                stats['successful_conversions'] += 1
            except Exception:
                stats['failed_conversions'] += 1

            stats['total_processed'] += 1

        return processed_files, class_names

    def _process_coco_json(self, data, image_files, stats):
        """Process COCO JSON format"""
        processed_files = []
        class_names = set()

        # Create mappings
        image_info = {img['id']: img for img in data['images']}
        category_info = {cat['id']: cat['name'] for cat in data['categories']}

        # Group annotations by image
        img_annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)

        for img_id, annotations in img_annotations.items():
            if img_id in image_info:
                img_info = image_info[img_id]
                img_filename = img_info['file_name']

                # Find corresponding image file
                img_path = None
                for full_path in image_files:
                    if os.path.basename(full_path) == img_filename:
                        img_path = full_path
                        break

                if img_path:
                    # Convert annotations to YOLO format
                    yolo_lines = []
                    for ann in annotations:
                        bbox = ann['bbox']  # [x, y, width, height]
                        img_width = img_info['width']
                        img_height = img_info['height']

                        # Convert to YOLO format (normalized center coordinates)
                        center_x = (bbox[0] + bbox[2] / 2) / img_width
                        center_y = (bbox[1] + bbox[3] / 2) / img_height
                        width = bbox[2] / img_width
                        height = bbox[3] / img_height

                        class_id = ann['category_id']
                        class_name = category_info[class_id]
                        class_names.add(class_name)

                        yolo_lines.append(f"{class_id} {center_x} {center_y} {width} {height}")

                        stats['class_distribution'][class_id] = stats['class_distribution'].get(class_id, 0) + 1

                    # Create temporary label file
                    temp_label = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
                    temp_label.write('\n'.join(yolo_lines))
                    temp_label.close()

                    processed_files.append((img_path, temp_label.name))

        return processed_files, class_names

    def _process_images_only(self, image_files, stats):
        """Process images without labels"""
        processed_files = []
        class_names = {"object"}

        for img_path in image_files:
            # Create empty label file
            temp_label = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_label.close()
            processed_files.append((img_path, temp_label.name))
            stats['empty_labels'] += 1
            stats['total_processed'] += 1

        return processed_files, class_names

    def _process_custom_format(self, dataset_path, image_files, stats):
        """Process custom format - fallback method"""
        # For now, treat as images only
        return self._process_images_only(image_files, stats)

    def _copy_files_to_split(self, file_pairs, img_dir, label_dir):
        """Copy files to train/val/test directories"""
        for img_path, label_path in file_pairs:
            img_name = os.path.basename(img_path)
            label_name = os.path.splitext(img_name)[0] + '.txt'

            # Copy image
            shutil.copy2(img_path, os.path.join(img_dir, img_name))

            # Copy label
            shutil.copy2(label_path, os.path.join(label_dir, label_name))


class YOLOModelTrainer:
    """Advanced YOLO model training with progress tracking and device detection"""

    def __init__(self):
        self.model = None
        self.training_results = None
        self.training_progress = {}

    def detect_device(self):
        """Automatically detect the best available device"""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count == 1:
                return '0'
            elif device_count > 1:
                return ','.join([str(i) for i in range(device_count)])
            else:
                return 'cpu'
        else:
            return 'cpu'

    def train_model(self, yaml_path: str, model_size='yolov8n', **kwargs):
        """Train YOLO model with comprehensive configuration and device detection"""

        # Model size mapping
        model_map = {
            'nano': 'yolov8n.pt',
            'small': 'yolov8s.pt',
            'medium': 'yolov8m.pt',
            'large': 'yolov8l.pt',
            'extra_large': 'yolov8x.pt'
        }

        base_model = model_map.get(model_size, 'yolov8n.pt')

        # Detect best available device if not specified
        auto_device = self.detect_device()
        selected_device = kwargs.get('device', auto_device)

        try:
            # Initialize model
            model = YOLO(base_model)

            # Training parameters
            train_params = {
                'data': yaml_path,
                'epochs': kwargs.get('epochs', 100),
                'imgsz': kwargs.get('img_size', 640),
                'batch': kwargs.get('batch_size', 16),
                'lr0': kwargs.get('learning_rate', 0.01),
                'lrf': kwargs.get('final_lr', 0.01),
                'momentum': kwargs.get('momentum', 0.937),
                'weight_decay': kwargs.get('weight_decay', 0.0005),
                'warmup_epochs': kwargs.get('warmup_epochs', 3),
                'warmup_momentum': kwargs.get('warmup_momentum', 0.8),
                'warmup_bias_lr': kwargs.get('warmup_bias_lr', 0.1),
                'box': kwargs.get('box_loss_gain', 0.05),
                'cls': kwargs.get('cls_loss_gain', 0.5),
                'dfl': kwargs.get('dfl_loss_gain', 1.5),
                'pose': kwargs.get('pose_loss_gain', 12.0),
                'kobj': kwargs.get('kobj_loss_gain', 1.0),
                'label_smoothing': kwargs.get('label_smoothing', 0.0),
                'nbs': kwargs.get('nominal_batch_size', 64),
                'overlap_mask': kwargs.get('overlap_mask', True),
                'mask_ratio': kwargs.get('mask_ratio', 4),
                'dropout': kwargs.get('dropout', 0.0),
                'val': kwargs.get('validate', True),
                'plots': kwargs.get('save_plots', True),
                'save': kwargs.get('save_model', True),
                'save_period': kwargs.get('save_period', 10),
                'cache': kwargs.get('cache_images', False),
                'device': selected_device,
                'workers': kwargs.get('workers', 8),
                'project': kwargs.get('project', 'runs/train'),
                'name': kwargs.get('experiment_name', 'custom_model'),
                'exist_ok': kwargs.get('exist_ok', False),
                'pretrained': kwargs.get('use_pretrained', True),
                'optimizer': kwargs.get('optimizer', 'auto'),
                'verbose': kwargs.get('verbose', True),
                'seed': kwargs.get('seed', 0),
                'deterministic': kwargs.get('deterministic', True),
                'single_cls': kwargs.get('single_class', False),
                'rect': kwargs.get('rectangular_training', False),
                'cos_lr': kwargs.get('cosine_lr', False),
                'close_mosaic': kwargs.get('close_mosaic', 10),
                'resume': kwargs.get('resume', False),
                'amp': kwargs.get('mixed_precision', True),
                'fraction': kwargs.get('dataset_fraction', 1.0),
                'profile': kwargs.get('profile', False),
                'patience': kwargs.get('early_stopping', 100)
            }

            # Adjust parameters for CPU training
            if selected_device == 'cpu':
                train_params['batch'] = min(train_params['batch'], 8)  # Reduce batch size for CPU
                train_params['workers'] = min(train_params['workers'], 4)  # Reduce workers for CPU
                train_params['amp'] = False  # Disable mixed precision for CPU

                # Display CPU warning
                st.warning(
                    "Training on CPU detected. This will be significantly slower than GPU training. Consider reducing epochs for testing.")

            # Start training
            results = model.train(**train_params)

            self.model = model
            self.training_results = results

            return results

        except Exception as e:
            st.error(f"Training failed: {str(e)}")

            # Additional debugging info
            st.error(
                f"Device info: torch.cuda.is_available()={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}")
            if torch.cuda.is_available():
                st.info(
                    f"Available CUDA devices: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")

            return None

    def get_training_metrics(self):
        """Extract training metrics from results"""
        if not self.training_results:
            return None

        try:
            results_dict = {}
            if hasattr(self.training_results, 'results_dict'):
                results_dict = self.training_results.results_dict

            return {
                'best_fitness': getattr(self.training_results, 'best_fitness', None),
                'save_dir': str(getattr(self.training_results, 'save_dir', '')),
                'metrics': results_dict
            }
        except Exception:
            return None


def create_training_dashboard(stats, metrics=None):
    """Create comprehensive training dashboard"""

    # Dataset Statistics
    st.subheader("Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", stats.get('total_processed', 0))
    with col2:
        st.metric("Training Images", stats.get('train_count', 0))
    with col3:
        st.metric("Validation Images", stats.get('val_count', 0))
    with col4:
        st.metric("Total Classes", stats.get('total_classes', 0))

    # Class distribution chart
    if stats.get('class_distribution'):
        st.subheader("Class Distribution")

        class_data = stats['class_distribution']
        fig = px.bar(
            x=list(class_data.keys()),
            y=list(class_data.values()),
            labels={'x': 'Class ID', 'y': 'Number of Instances'},
            title="Distribution of Classes in Dataset"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Conversion statistics
    if stats.get('successful_conversions'):
        st.subheader("Conversion Statistics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Successful Conversions", stats.get('successful_conversions', 0))
        with col2:
            st.metric("Failed Conversions", stats.get('failed_conversions', 0))
        with col3:
            st.metric("Empty Labels", stats.get('empty_labels', 0))

    # Training metrics (if available)
    if metrics:
        st.subheader("Training Metrics")

        training_metrics = metrics.get('metrics', {})
        if training_metrics:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if 'train/box_loss' in training_metrics:
                    st.metric("Train Box Loss", f"{training_metrics['train/box_loss']:.4f}")
            with col2:
                if 'train/cls_loss' in training_metrics:
                    st.metric("Train Cls Loss", f"{training_metrics['train/cls_loss']:.4f}")
            with col3:
                if 'val/box_loss' in training_metrics:
                    st.metric("Val Box Loss", f"{training_metrics['val/box_loss']:.4f}")
            with col4:
                if 'val/cls_loss' in training_metrics:
                    st.metric("Val Cls Loss", f"{training_metrics['val/cls_loss']:.4f}")


def main():
    st.set_page_config(
        page_title="YOLO Custom Model Trainer",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 Custom YOLO Model Training System")
    st.markdown("**Advanced dataset processing and custom model training with comprehensive analytics**")

    # Initialize components
    if 'dataset_processor' not in st.session_state:
        st.session_state.dataset_processor = SmartDatasetProcessor()
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = YOLOModelTrainer()
    if 'processed_dataset' not in st.session_state:
        st.session_state.processed_dataset = None

    # Display device information
    st.sidebar.header("System Information")
    available_devices = get_available_devices()

    if torch.cuda.is_available():
        st.sidebar.success(f"✅ GPU Available: {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            st.sidebar.info(f"GPU {i}: {gpu_name}")
    else:
        st.sidebar.warning("⚠️ No GPU detected - CPU training only")
        st.sidebar.info("For faster training, consider using a GPU-enabled environment")

    # Sidebar for dataset input
    with st.sidebar:
        st.header("📁 Dataset Configuration")

        # Dataset path input
        dataset_path = st.text_input(
            "Dataset Path:",
            placeholder=r"C:\path\to\your\dataset",
            help="Path to folder containing images and labels"
        )

        # Dataset upload option
        st.subheader("Or Upload Dataset")
        uploaded_zip = st.file_uploader(
            "Upload dataset ZIP file",
            type=['zip'],
            help="ZIP file containing images and labels"
        )

        if uploaded_zip:
            # Extract uploaded dataset
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "dataset.zip")

            with open(zip_path, 'wb') as f:
                f.write(uploaded_zip.read())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            dataset_path = temp_dir
            st.success("✅ Dataset uploaded successfully!")

    # Main content
    if dataset_path and os.path.exists(dataset_path):

        # Dataset analysis tab
        tab1, tab2, tab3 = st.tabs(["📊 Dataset Analysis", "⚙️ Processing & Training", "📈 Results"])

        with tab1:
            st.header("Dataset Analysis")

            if st.button("🔍 Analyze Dataset", type="primary"):
                with st.spinner("Analyzing dataset..."):
                    analysis = st.session_state.dataset_processor.analyze_dataset(dataset_path)

                st.subheader("📈 Analysis Results")

                # Basic stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Images", analysis['total_images'])
                with col2:
                    st.metric("Total Labels", analysis['total_labels'])
                with col3:
                    st.metric("Detected Format", analysis['format'])
                with col4:
                    st.metric("Issues Found", len(analysis['issues']))

                # Image size distribution
                if analysis['image_sizes']:
                    st.subheader("📐 Image Size Distribution")

                    widths = [size[0] for size in analysis['image_sizes']]
                    heights = [size[1] for size in analysis['image_sizes']]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=widths, y=heights, mode='markers',
                        marker=dict(size=8, opacity=0.6),
                        name='Image Dimensions'
                    ))
                    fig.update_layout(
                        title="Image Dimensions Scatter Plot",
                        xaxis_title="Width (pixels)",
                        yaxis_title="Height (pixels)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Class distribution
                if analysis['class_distribution']:
                    st.subheader("🏷️ Class Distribution")

                    class_data = analysis['class_distribution']
                    fig = px.pie(
                        values=list(class_data.values()),
                        names=[f"Class {k}" for k in class_data.keys()],
                        title="Class Distribution in Dataset"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Issues
                if analysis['issues']:
                    st.subheader("⚠️ Issues Found")
                    for issue in analysis['issues']:
                        st.warning(issue)

        with tab2:
            st.header("Dataset Processing & Model Training")

            # Dataset processing section
            st.subheader("🔄 Dataset Processing")

            col1, col2 = st.columns(2)
            with col1:
                train_split = st.slider("Training Split", 0.6, 0.9, 0.8, 0.05)
                val_split = st.slider("Validation Split", 0.1, 0.3, 0.15, 0.05)

            with col2:
                st.write(f"**Test Split:** {1.0 - train_split - val_split:.2f}")
                st.write(f"**Training:** {train_split * 100:.0f}%")
                st.write(f"**Validation:** {val_split * 100:.0f}%")
                st.write(f"**Testing:** {(1.0 - train_split - val_split) * 100:.0f}%")

            if st.button("🔄 Process Dataset", type="primary"):
                output_dir = tempfile.mkdtemp(prefix="yolo_processed_")

                try:
                    with st.spinner("Processing dataset to YOLO format..."):
                        yaml_path, class_names, stats = st.session_state.dataset_processor.convert_to_yolo_format(
                            dataset_path, output_dir, train_split, val_split
                        )

                    st.success("✅ Dataset processed successfully!")

                    # Store processed dataset info
                    st.session_state.processed_dataset = {
                        'yaml_path': yaml_path,
                        'class_names': class_names,
                        'stats': stats,
                        'output_dir': output_dir
                    }

                    # Display processing results
                    create_training_dashboard(stats)

                except Exception as e:
                    st.error(f"❌ Dataset processing failed: {str(e)}")

            # Model training section
            if st.session_state.processed_dataset:
                st.subheader("🏋️ Model Training Configuration")

                # Hardware Configuration
                with st.expander("🖥️ Hardware Configuration"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Device Selection")

                        # Auto-select best device
                        default_device = 'cpu'
                        if len(available_devices) > 1:  # If GPU available
                            default_device = available_devices[1]  # First GPU

                        selected_device = st.selectbox(
                            "Training Device",
                            available_devices,
                            index=available_devices.index(default_device),
                            help="Choose CPU for compatibility or GPU for speed"
                        )

                        # Show device info
                        if selected_device == 'cpu':
                            st.info("CPU training is slower but more compatible")
                        elif 'cuda' in selected_device:
                            st.success("GPU training will be much faster")
                        elif 'multi-gpu' in selected_device:
                            st.success("Multi-GPU training for maximum speed")

                    with col2:
                        st.subheader("Performance Settings")

                        # Adjust default batch size based on device
                        if selected_device == 'cpu':
                            default_batch = 4
                            max_batch = 16
                            batch_help = "Smaller batches recommended for CPU"
                            default_workers = 2
                            max_workers = 8
                        else:
                            default_batch = 16
                            max_batch = 64
                            batch_help = "Larger batches possible with GPU"
                            default_workers = 8
                            max_workers = 16

                        batch_size = st.number_input(
                            "Batch Size",
                            1, max_batch,
                            default_batch,
                            help=batch_help
                        )

                        workers = st.number_input("Data Workers", 0, max_workers, default_workers)

                # Model configuration
                col1, col2, col3 = st.columns(3)

                with col1:
                    model_size = st.selectbox(
                        "Model Size",
                        ['nano', 'small', 'medium', 'large', 'extra_large'],
                        help="Nano: fastest, Extra Large: most accurate"
                    )
                    epochs = st.number_input("Epochs", 10, 1000, 100)
                    img_size = st.selectbox("Image Size", [320, 416, 640, 832, 1280], index=2)

                with col2:
                    learning_rate = st.number_input("Learning Rate", 0.001, 0.1, 0.01, format="%.4f")
                    momentum = st.number_input("Momentum", 0.8, 0.99, 0.937, format="%.3f")
                    weight_decay = st.number_input("Weight Decay", 0.0, 0.01, 0.0005, format="%.4f")

                with col3:
                    patience = st.number_input("Early Stopping Patience", 10, 200, 50)
                    save_period = st.number_input("Save Every N Epochs", 1, 50, 10)

                # Advanced settings
                with st.expander("🔧 Advanced Training Settings"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.subheader("Optimization")
                        optimizer = st.selectbox("Optimizer", ['auto', 'SGD', 'Adam', 'AdamW'])
                        cos_lr = st.checkbox("Cosine Learning Rate", False)
                        mixed_precision = st.checkbox("Mixed Precision (AMP)",
                                                      True if selected_device != 'cpu' else False)

                    with col2:
                        st.subheader("Augmentation")
                        rect_training = st.checkbox("Rectangular Training", False)
                        close_mosaic = st.number_input("Close Mosaic (epochs)", 0, 50, 10)
                        cache_images = st.checkbox("Cache Images", False)

                    with col3:
                        st.subheader("Loss Configuration")
                        box_loss_gain = st.number_input("Box Loss Gain", 0.01, 10.0, 0.05, format="%.3f")
                        cls_loss_gain = st.number_input("Class Loss Gain", 0.1, 10.0, 0.5, format="%.3f")
                        label_smoothing = st.number_input("Label Smoothing", 0.0, 0.3, 0.0, format="%.3f")

                # Training execution
                st.subheader("🚀 Start Training")

                # Estimated training time
                estimated_time = (epochs * st.session_state.processed_dataset['stats']['train_count']) / (
                            batch_size * 100)  # rough estimate
                if selected_device == 'cpu':
                    estimated_time *= 20  # CPU is much slower

                st.info(f"🕐 Estimated training time: {estimated_time:.1f} hours (depends on hardware)")

                col1, col2 = st.columns([3, 1])
                with col1:
                    experiment_name = st.text_input("Experiment Name",
                                                    f"custom_model_{datetime.now().strftime('%Y%m%d_%H%M')}")
                with col2:
                    st.write("**Dataset Ready:**")
                    st.success(f"✅ {st.session_state.processed_dataset['stats']['total_classes']} classes")

                if st.button("🚀 Start Training", type="primary", use_container_width=True):

                    # Prepare training parameters
                    training_params = {
                        'epochs': epochs,
                        'img_size': img_size,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'momentum': momentum,
                        'weight_decay': weight_decay,
                        'early_stopping': patience,
                        'save_period': save_period,
                        'optimizer': optimizer,
                        'cosine_lr': cos_lr,
                        'mixed_precision': mixed_precision,
                        'rectangular_training': rect_training,
                        'close_mosaic': close_mosaic,
                        'cache_images': cache_images,
                        'box_loss_gain': box_loss_gain,
                        'cls_loss_gain': cls_loss_gain,
                        'label_smoothing': label_smoothing,
                        'experiment_name': experiment_name,
                        'device': format_device_selection(selected_device),
                        'workers': workers,
                        'save_plots': True,
                        'verbose': True
                    }

                    # Training progress tracking
                    progress_container = st.container()
                    with progress_container:
                        st.subheader("🏋️ Training Progress")

                        # Progress metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            epoch_metric = st.empty()
                        with col2:
                            loss_metric = st.empty()
                        with col3:
                            map_metric = st.empty()
                        with col4:
                            time_metric = st.empty()

                        # Progress bar and status
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Training log
                        log_container = st.expander("📋 Training Log", expanded=False)
                        training_log = log_container.empty()

                    try:
                        # Start training
                        start_time = time.time()
                        status_text.text("🚀 Initializing training...")

                        results = st.session_state.model_trainer.train_model(
                            st.session_state.processed_dataset['yaml_path'],
                            model_size,
                            **training_params
                        )

                        if results:
                            training_time = time.time() - start_time

                            # Update progress to 100%
                            progress_bar.progress(1.0)
                            status_text.text("✅ Training completed successfully!")

                            # Get training metrics
                            metrics = st.session_state.model_trainer.get_training_metrics()

                            # Success message
                            st.success("🎉 Model training completed successfully!")
                            st.balloons()

                            # Store results for display in results tab
                            st.session_state.training_completed = {
                                'results': results,
                                'metrics': metrics,
                                'training_time': training_time,
                                'model_path': metrics['save_dir'] if metrics else None
                            }

                            # Display immediate results
                            st.subheader("📊 Training Summary")

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Training Time", f"{training_time / 3600:.1f} hours")
                            with col2:
                                st.metric("Total Epochs", epochs)
                            with col3:
                                if metrics and 'best_fitness' in metrics and metrics['best_fitness']:
                                    st.metric("Best Fitness", f"{metrics['best_fitness']:.4f}")
                                else:
                                    st.metric("Status", "Completed")
                            with col4:
                                st.metric("Model Size", model_size.title())

                            # Model save location
                            if metrics and metrics.get('save_dir'):
                                st.info(f"📁 **Model saved to:** `{metrics['save_dir']}`")

                                # Check for specific model files
                                best_model = os.path.join(metrics['save_dir'], 'weights', 'best.pt')
                                last_model = os.path.join(metrics['save_dir'], 'weights', 'last.pt')

                                col1, col2 = st.columns(2)
                                with col1:
                                    if os.path.exists(best_model):
                                        st.success("✅ Best model saved")
                                        with open(best_model, 'rb') as f:
                                            st.download_button(
                                                "⬇️ Download Best Model",
                                                f.read(),
                                                file_name=f"{experiment_name}_best.pt",
                                                mime="application/octet-stream"
                                            )
                                with col2:
                                    if os.path.exists(last_model):
                                        st.success("✅ Last model saved")
                                        with open(last_model, 'rb') as f:
                                            st.download_button(
                                                "⬇️ Download Last Model",
                                                f.read(),
                                                file_name=f"{experiment_name}_last.pt",
                                                mime="application/octet-stream"
                                            )

                        else:
                            st.error("❌ Training failed. Please check your dataset and parameters.")

                    except Exception as e:
                        st.error(f"❌ Training error: {str(e)}")
                        status_text.text("❌ Training failed")

        with tab3:
            st.header("Training Results & Analysis")

            if hasattr(st.session_state, 'training_completed'):
                results_data = st.session_state.training_completed

                # Training metrics dashboard
                st.subheader("📈 Training Metrics")

                if results_data['metrics']:
                    metrics = results_data['metrics']['metrics']

                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if 'train/box_loss' in metrics:
                            st.metric(
                                "Final Train Box Loss",
                                f"{metrics['train/box_loss']:.4f}",
                                help="Lower is better"
                            )
                    with col2:
                        if 'train/cls_loss' in metrics:
                            st.metric(
                                "Final Train Cls Loss",
                                f"{metrics['train/cls_loss']:.4f}",
                                help="Lower is better"
                            )
                    with col3:
                        if 'val/box_loss' in metrics:
                            st.metric(
                                "Final Val Box Loss",
                                f"{metrics['val/box_loss']:.4f}",
                                help="Lower is better"
                            )
                    with col4:
                        if 'val/cls_loss' in metrics:
                            st.metric(
                                "Final Val Cls Loss",
                                f"{metrics['val/cls_loss']:.4f}",
                                help="Lower is better"
                            )

                # Model performance metrics
                st.subheader("🎯 Model Performance")

                performance_cols = st.columns(4)
                performance_metrics = ['precision', 'recall', 'mAP50', 'mAP50-95']

                for i, metric in enumerate(performance_metrics):
                    with performance_cols[i]:
                        metric_key = f'metrics/{metric}(B)' if metric.startswith('mAP') else f'metrics/{metric}(B)'
                        if metric_key in metrics:
                            st.metric(
                                metric.upper(),
                                f"{metrics[metric_key]:.4f}",
                                help=f"Higher is better for {metric}"
                            )

                # Training summary
                st.subheader("📋 Training Summary")

                summary_data = {
                    "Training Duration": f"{results_data['training_time'] / 3600:.2f} hours",
                    "Model Architecture": st.session_state.processed_dataset['stats']['dataset_format'],
                    "Dataset Classes": len(st.session_state.processed_dataset['class_names']),
                    "Training Images": st.session_state.processed_dataset['stats']['train_count'],
                    "Validation Images": st.session_state.processed_dataset['stats']['val_count'],
                    "Model Save Path": results_data['model_path'] or "Not available"
                }

                for key, value in summary_data.items():
                    st.write(f"**{key}:** {value}")

                # Model export options
                st.subheader("📤 Model Export & Deployment")

                if results_data['model_path']:
                    best_model_path = os.path.join(results_data['model_path'], 'weights', 'best.pt')

                    if os.path.exists(best_model_path):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            if st.button("Export to ONNX"):
                                try:
                                    model = YOLO(best_model_path)
                                    export_path = model.export(format='onnx')
                                    st.success(f"✅ ONNX export completed: {export_path}")
                                except Exception as e:
                                    st.error(f"ONNX export failed: {e}")

                        with col2:
                            if st.button("Export to TensorRT"):
                                try:
                                    model = YOLO(best_model_path)
                                    export_path = model.export(format='engine')
                                    st.success(f"✅ TensorRT export completed: {export_path}")
                                except Exception as e:
                                    st.error(f"TensorRT export failed: {e}")

                        with col3:
                            if st.button("Export to CoreML"):
                                try:
                                    model = YOLO(best_model_path)
                                    export_path = model.export(format='coreml')
                                    st.success(f"✅ CoreML export completed: {export_path}")
                                except Exception as e:
                                    st.error(f"CoreML export failed: {e}")

                # Quick model validation
                st.subheader("🔬 Quick Model Validation")

                uploaded_test_image = st.file_uploader(
                    "Test your trained model with an image",
                    type=['jpg', 'jpeg', 'png'],
                    key="test_image"
                )

                if uploaded_test_image and results_data['model_path']:
                    test_image = Image.open(uploaded_test_image)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Original Image")
                        st.image(test_image, use_column_width=True)

                    with col2:
                        st.subheader("Detection Results")

                        try:
                            best_model_path = os.path.join(results_data['model_path'], 'weights', 'best.pt')
                            if os.path.exists(best_model_path):
                                model = YOLO(best_model_path)

                                # Run inference
                                results = model(np.array(test_image))

                                # Display results
                                if results:
                                    annotated_img = results[0].plot()
                                    st.image(annotated_img, use_column_width=True)

                                    # Detection details
                                    if results[0].boxes is not None:
                                        st.write("**Detected Objects:**")
                                        for i, box in enumerate(results[0].boxes):
                                            conf = box.conf[0].cpu().numpy()
                                            cls = int(box.cls[0].cpu().numpy())
                                            class_name = model.names[cls]
                                            st.write(f"• {class_name}: {conf:.2f}")
                                    else:
                                        st.info("No objects detected")

                        except Exception as e:
                            st.error(f"Inference failed: {e}")

            else:
                st.info("👆 Complete training in the 'Processing & Training' tab to see results here.")

                # Display dataset processing results if available
                if st.session_state.processed_dataset:
                    st.subheader("📊 Dataset Processing Results")
                    create_training_dashboard(st.session_state.processed_dataset['stats'])

    else:
        # Welcome screen
        st.header("👋 Welcome to Custom YOLO Trainer")

        st.markdown("""
        ### 🎯 What this system does:

        1. **Smart Dataset Analysis** 📊
           - Automatically detects dataset format (YOLO, COCO, custom JSON, etc.)
           - Analyzes image dimensions, class distribution, and data quality
           - Identifies potential issues in your dataset

        2. **Intelligent Data Processing** 🔄
           - Converts any supported format to YOLO format automatically
           - Splits data into train/validation/test sets
           - Generates all necessary configuration files

        3. **Advanced Model Training** 🏋️
           - Multiple model sizes from nano to extra-large
           - Comprehensive hyperparameter tuning
           - Real-time training progress monitoring
           - Advanced optimization techniques

        4. **Complete Analysis & Export** 📈
           - Detailed training metrics and performance analysis
           - Model validation and testing tools
           - Multiple export formats (ONNX, TensorRT, CoreML)
           - Ready-to-deploy models

        ### 📂 Supported Dataset Formats:
        - **YOLO Format**: Images + TXT files with normalized coordinates
        - **COCO JSON**: Standard COCO annotation format
        - **Custom JSON**: Your custom annotation format
        - **Images Only**: Just images (creates template for manual annotation)

        ### 🚀 Get Started:
        1. Enter your dataset path in the sidebar, or upload a ZIP file
        2. The system will automatically analyze and process your data
        3. Configure training parameters and start training
        4. Monitor progress and export your trained model

        ---
        **💡 Tip:** For best results, ensure your dataset has:
        - At least 100+ images per class
        - Consistent image quality and lighting
        - Balanced class distribution
        - Accurate annotations
        """)

        # Feature highlights
        st.subheader("✨ Key Features")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            **🔍 Smart Analysis**
            - Auto-format detection
            - Data quality assessment
            - Performance predictions
            - Issue identification
            """)

        with col2:
            st.markdown("""
            **⚙️ Advanced Training**
            - Multiple model architectures
            - Hyperparameter optimization
            - Early stopping & scheduling
            - Mixed precision training
            """)

        with col3:
            st.markdown("""
            **📊 Comprehensive Analytics**
            - Real-time metrics
            - Training visualizations
            - Model validation
            - Export capabilities
            """)


if __name__ == "__main__":
    main()