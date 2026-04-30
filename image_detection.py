from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import os
import warnings
import time
import numpy as np
import base64
from io import BytesIO
import json

import re
import hashlib
import secrets
import uuid
import functools
from difflib import SequenceMatcher

warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

app = Flask(__name__)

# 启用CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# 配置
DATABASE_FOLDER = r"D:\code\Github_dinov2\phototest"
UPLOAD_FOLDER = r"D:\code\Github_dinov2\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 全局变量
model = None
device = None
transform = None
feature_matrix = None
feature_matrix_original = None
color_feature_matrix = None
texture_feature_matrix = None
image_paths_original = []
system_ready = False
# 缓存文件路径（两个模式共用一个缓存）
CACHE_FILE = r"D:\code\Github_dinov2\feature_cache_subimages.pth"

# 搜索结果分页缓存 key: search_id, value: {results: [...], time: ..., mode: ..., ...}
search_results_cache = {}
SEARCH_CACHE_MAX_AGE = 600  # 搜索结果缓存10分钟

# 商品信息存储
PRODUCT_INFO_FILE = r"D:\code\Github_dinov2\product_info.json"
product_info_db = {}  # key: filename, value: {name, material, craft, pattern, description, records: [...]}

# ============================================================
# 用户认证系统
# ============================================================
USER_DB_FILE = r"D:\code\Github_dinov2\users.json"
users_db = {}       # key: username, value: {password_hash, role, max_results, created_time, ...}
sessions_db = {}    # key: token, value: {username, login_time}

def hash_password(password):
    """对密码进行SHA256哈希"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def load_users():
    """从JSON文件加载用户数据"""
    global users_db
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, 'r', encoding='utf-8') as f:
                users_db = json.load(f)
            print(f"已加载 {len(users_db)} 个用户账户")
        except Exception as e:
            print(f"加载用户数据失败: {e}")
            users_db = {}
    
    # 确保管理员账户存在
    if 'admin' not in users_db:
        users_db['admin'] = {
            'password_hash': hash_password('admin123'),
            'role': 'admin',
            'max_results': 9999,
            'can_search': True,
            'can_view_products': True,
            'can_edit_products': True,
            'can_manage_records': True,
            'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'display_name': '系统管理员'
        }
        save_users()
        print("已创建默认管理员账户 (admin / admin123)")

def save_users():
    """保存用户数据到JSON文件"""
    try:
        with open(USER_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存用户数据失败: {e}")

def get_current_user():
    """从请求头中的token获取当前登录用户"""
    token = request.headers.get('X-Auth-Token', '') or request.cookies.get('auth_token', '')
    if token and token in sessions_db:
        username = sessions_db[token]['username']
        if username in users_db:
            return username, users_db[username]
    return None, None

def login_required(f):
    """登录验证装饰器"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        username, user = get_current_user()
        if not username:
            return jsonify({'error': '请先登录', 'need_login': True}), 401
        request.current_user = username
        request.current_user_info = user
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    """管理员权限验证装饰器"""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        username, user = get_current_user()
        if not username:
            return jsonify({'error': '请先登录', 'need_login': True}), 401
        if user.get('role') != 'admin':
            return jsonify({'error': '需要管理员权限'}), 403
        request.current_user = username
        request.current_user_info = user
        return f(*args, **kwargs)
    return decorated

def load_product_info():
    """从JSON文件加载商品信息"""
    global product_info_db
    if os.path.exists(PRODUCT_INFO_FILE):
        try:
            with open(PRODUCT_INFO_FILE, 'r', encoding='utf-8') as f:
                product_info_db = json.load(f)
            print(f"已加载 {len(product_info_db)} 条商品信息")
        except Exception as e:
            print(f"加载商品信息失败: {e}")
            product_info_db = {}

def save_product_info():
    """保存商品信息到JSON文件"""
    try:
        with open(PRODUCT_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(product_info_db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存商品信息失败: {e}")

def initialize_product_info():
    """扫描数据库文件夹，初始化商品信息（不覆盖已有信息）"""
    global product_info_db
    load_product_info()
    
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    updated = False
    
    for root, dirs, files in os.walk(DATABASE_FOLDER):
        for filename in files:
            if filename.lower().endswith(supported_ext):
                filepath = os.path.join(root, filename)
                dir_name = os.path.basename(os.path.dirname(filepath))
                
                if filename not in product_info_db:
                    # 根据子文件夹名称设置花纹（all文件夹除外）
                    pattern = ''
                    if dir_name != 'all' and dir_name != os.path.basename(DATABASE_FOLDER):
                        pattern = dir_name
                    
                    product_info_db[filename] = {
                        'name': filename,
                        'material': '',
                        'craft': '',
                        'pattern': pattern,
                        'description': '',
                        'path': filepath,
                        'order_records': [],
                        'recommend_records': []
                    }
                    updated = True
                else:
                    # 更新路径（可能变化）和花纹标记
                    product_info_db[filename]['path'] = filepath
                    # 如果花纹为空且在子文件夹中，自动标记
                    if not product_info_db[filename].get('pattern') and dir_name != 'all' and dir_name != os.path.basename(DATABASE_FOLDER):
                        product_info_db[filename]['pattern'] = dir_name
                        updated = True
    
    if updated:
        save_product_info()
    print(f"商品信息初始化完成，共 {len(product_info_db)} 个商品")

def extract_price_from_description(description):
    """从信息描述中提取价格，匹配 'xx元' 格式"""
    if not description:
        return None
    match = re.search(r'(\d+(?:\.\d+)?)\s*元', description)
    if match:
        return float(match.group(1))
    return None

def text_match_score(query, product_info):
    """计算查询文本与商品所有信息字段的匹配度"""
    if not query:
        return 0.0
    
    query_lower = query.lower().strip()
    
    # 收集所有可搜索字段
    fields = [
        product_info.get('name', ''),
        product_info.get('material', ''),
        product_info.get('craft', ''),
        product_info.get('pattern', ''),
        product_info.get('description', ''),
    ]
    
    # 合并为一个文本
    all_text = ' '.join(f for f in fields if f).lower()
    
    if not all_text:
        return 0.0
    
    # 完全包含匹配 - 高分
    score = 0.0
    
    # 拆分查询为关键词
    query_words = query_lower.split()
    
    for word in query_words:
        if word in all_text:
            score += 1.0
        else:
            # 使用模糊匹配
            best_ratio = 0.0
            for field in fields:
                if field:
                    ratio = SequenceMatcher(None, word, field.lower()).ratio()
                    best_ratio = max(best_ratio, ratio)
                    # 检查子串
                    for i in range(len(field) - len(word) + 1):
                        substr = field[i:i+len(word)].lower()
                        r = SequenceMatcher(None, word, substr).ratio()
                        best_ratio = max(best_ratio, r)
            score += best_ratio * 0.5
    
    # 归一化
    if len(query_words) > 0:
        score = score / len(query_words)
    
    return score

# ============================================================
# 1. 初始化系统
# ============================================================
def initialize_system():
    global model, device, transform, feature_matrix, image_paths, system_ready
    global color_feature_matrix, texture_feature_matrix, feature_matrix_original, image_paths_original

    try:
        print("正在初始化万斯图像检索系统...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print("正在加载模型...")
        local_repo_path = r"D:\code\Github_dinov2\dinov2-main"
        model = torch.hub.load(local_repo_path, 'dinov2_vits14', source='local', pretrained=False)

        weight_path = r"D:\code\Github_dinov2\dinov2-main\dinov2_vits14_pretrain.pth"
        state_dict = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("模型加载成功")

        # 加载缓存
        print("\n=== 加载缓存 ===")
        if os.path.exists(CACHE_FILE):
            print("发现缓存文件，正在加载...")
            try:
                cache_data = torch.load(CACHE_FILE, map_location="cpu")
                feature_matrix = cache_data['feature_matrix']
                image_paths = cache_data['image_paths']
                color_feature_matrix = cache_data.get('color_feature_matrix')
                texture_feature_matrix = cache_data.get('texture_feature_matrix')
                print(f"缓存加载成功，包含 {len(image_paths)} 个子图")
            except Exception as e:
                print(f"缓存加载失败: {e}，将重新构建")
                feature_matrix, image_paths, color_feature_matrix, texture_feature_matrix = build_feature_database_subimages(DATABASE_FOLDER, model, transform, device)
                save_cache(CACHE_FILE, feature_matrix, image_paths, color_feature_matrix, texture_feature_matrix)
        else:
            print("未发现缓存，正在构建...")
            feature_matrix, image_paths, color_feature_matrix, texture_feature_matrix = build_feature_database_subimages(DATABASE_FOLDER, model, transform, device)
            save_cache(CACHE_FILE, feature_matrix, image_paths, color_feature_matrix, texture_feature_matrix)
        
        # 整体模式复用综合模式的特征矩阵，通过调整权重实现
        feature_matrix_original = feature_matrix
        image_paths_original = image_paths
        
        system_ready = True
        print("\n系统初始化完成！")
        print(f"  - 综合模式: {len(image_paths)} 个子图 (DINOv2 + 颜色 + 纹理)")
        print(f"  - 整体模式: 共用综合模式缓存，通过权重调整实现")

    except Exception as e:
        print(f"系统初始化失败: {e}")
        import traceback
        traceback.print_exc()

def save_cache(cache_file, fm, ip, cfm, tfm):
    print("正在保存综合模式缓存...")
    cache_data = {
        'feature_matrix': fm.cpu() if fm is not None else None,
        'image_paths': ip,
        'color_feature_matrix': cfm,
        'texture_feature_matrix': tfm,
        'timestamp': time.time(),
        'database_folder': os.path.abspath(DATABASE_FOLDER)
    }
    torch.save(cache_data, cache_file)
    print("综合模式缓存保存成功")

def save_cache_original(fm, ip):
    print("正在保存整体模式缓存...")
    cache_data = {
        'feature_matrix': fm.cpu() if fm is not None else None,
        'image_paths': ip,
        'timestamp': time.time(),
        'database_folder': os.path.abspath(DATABASE_FOLDER)
    }
    torch.save(cache_data, CACHE_FILE_ORIGINAL)
    print("整体模式缓存保存成功")

def build_feature_database_original(image_folder, model, transform, device):
    """构建仅包含DINOv2特征的整体模式数据库"""
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    original_image_paths = []

    # 递归扫描所有子文件夹
    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if filename.lower().endswith(supported_ext):
                original_image_paths.append(os.path.join(root, filename))

    if len(original_image_paths) == 0:
        print(f"错误：{image_folder} 中没有找到图片！")
        return None, []

    print(f"找到 {len(original_image_paths)} 张原图，将生成 {len(original_image_paths) * 21} 个子图...")

    features_list = []
    subimage_info_list = []

    for i, original_path in enumerate(original_image_paths):
        try:
            print(f"  [{i+1}/{len(original_image_paths)}] 处理: {os.path.basename(original_path)}")
            
            img_cv = cv2.imdecode(np.fromfile(original_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_cv is None:
                continue

            regions_info = get_multiscale_regions_with_info(img_cv, original_path)

            batch_tensors = []
            batch_info = []
            
            for region_info in regions_info:
                pil_img = Image.fromarray(cv2.cvtColor(region_info['image'], cv2.COLOR_BGR2RGB))
                tensor = transform(pil_img)
                batch_tensors.append(tensor)
                batch_info.append({
                    'original_path': region_info['original_path'],
                    'crop_type': region_info['crop_type'],
                    'coords': region_info['coords'],
                    'img_size': region_info['img_size']
                })

            batch_tensor = torch.stack(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                batch_features = model(batch_tensor)
            batch_features = F.normalize(batch_features, p=2, dim=1)

            for j, (feature, info) in enumerate(zip(batch_features, batch_info)):
                features_list.append(feature.unsqueeze(0))
                subimage_info_list.append(info)

        except Exception as e:
            print(f"    处理失败 {os.path.basename(original_path)}: {e}")

    if not features_list:
        print("错误：没有成功提取任何特征！")
        return None, []

    feature_matrix = torch.cat(features_list, dim=0)
    print(f"整体模式特征数据库构建完成！共 {len(subimage_info_list)} 个子图")
    
    return feature_matrix, subimage_info_list

# ============================================================
# 2. 多尺度裁剪函数（返回裁剪信息和坐标）
# ============================================================
def get_multiscale_regions_with_info(img_cv, original_path):
    """
    返回所有裁剪区域的信息：
    每个元素包含：(裁剪后的图像, 裁剪类型, 裁剪坐标, 原图路径, 原图尺寸)
    """
    h, w = img_cv.shape[:2]
    regions_info = []

    # 全图 1 块
    regions_info.append({
        'image': img_cv,
        'crop_type': 'full',
        'coords': (0, 0, w, h),
        'original_path': original_path,
        'img_size': (w, h)
    })

    # 2×2 网格 4 块
    for r in range(2):
        for c in range(2):
            y1 = r * h // 2
            y2 = h if r == 1 else (r + 1) * h // 2
            x1 = c * w // 2
            x2 = w if c == 1 else (c + 1) * w // 2
            regions_info.append({
                'image': img_cv[y1:y2, x1:x2],
                'crop_type': f'2x2_{r}_{c}',
                'coords': (x1, y1, x2, y2),
                'original_path': original_path,
                'img_size': (w, h)
            })

    # 4×4 网格 16 块
    for r in range(4):
        for c in range(4):
            y1 = r * h // 4
            y2 = h if r == 3 else (r + 1) * h // 4
            x1 = c * w // 4
            x2 = w if c == 3 else (c + 1) * w // 4
            regions_info.append({
                'image': img_cv[y1:y2, x1:x2],
                'crop_type': f'4x4_{r}_{c}',
                'coords': (x1, y1, x2, y2),
                'original_path': original_path,
                'img_size': (w, h)
            })

    return regions_info  # 共21个元素

# ============================================================
# 3. 特征提取
# ============================================================
def extract_feature_from_region(region_info, model, transform, device):
    """从裁剪区域提取特征"""
    region_img = region_info['image']
    
    # 转换为PIL图像
    pil_img = Image.fromarray(cv2.cvtColor(region_img, cv2.COLOR_BGR2RGB))
    
    # 预处理
    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # 提取特征
    with torch.no_grad():
        feature = model(img_tensor)
    
    # 归一化
    feature = F.normalize(feature, p=2, dim=1)
    return feature

def extract_color_histogram(img_cv, bins=32):
    """提取颜色直方图特征（RGB三通道）"""
    hist_features = []
    for i in range(3):
        hist = cv2.calcHist([img_cv], [i], None, [bins], [0, 256])
        hist = hist.flatten()
        hist = hist / (hist.sum() + 1e-7)
        hist_features.extend(hist)
    return np.array(hist_features, dtype=np.float32)

def create_gabor_kernels(ksize=21, num_theta=8, num_sigma=3):
    """创建Gabor滤波器核"""
    kernels = []
    for theta_idx in range(num_theta):
        theta = theta_idx * np.pi / num_theta
        for sigma in [num_sigma]:
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F
            )
            kernels.append(kernel)
    return kernels

def extract_gabor_texture(img_cv, kernels):
    """提取Gabor纹理特征"""
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv
    
    texture_features = []
    for kernel in kernels:
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        texture_features.append(filtered.mean())
        texture_features.append(filtered.std())
    
    texture_features = np.array(texture_features, dtype=np.float32)
    norm = np.linalg.norm(texture_features)
    if norm > 0:
        texture_features = texture_features / norm
    return texture_features

# ============================================================
# 4. 构建特征数据库（子图模式）
# ============================================================
def extract_attributes_from_path(image_path):
    """
    从商品信息数据库获取属性信息
    如果数据库中没有，则从路径提取基本信息
    """
    filename = os.path.basename(image_path)
    
    # 先从商品信息数据库获取
    if filename in product_info_db:
        info = product_info_db[filename]
        price = extract_price_from_description(info.get('description', ''))
        return {
            'attribute': info.get('pattern', ''),
            'material': info.get('material', ''),
            'craft': info.get('craft', ''),
            'pattern': info.get('pattern', ''),
            'price': price,
            'description': info.get('description', ''),
            'path': image_path
        }
    
    # 提取文件夹名称作为花纹
    dir_name = os.path.basename(os.path.dirname(image_path))
    
    # 排除不计算属性的文件夹
    exclude_folders = ['all', '其他']
    pattern = ''
    if dir_name not in exclude_folders and dir_name != os.path.basename(DATABASE_FOLDER):
        pattern = dir_name
    
    return {
        'attribute': pattern,
        'material': '',
        'craft': '',
        'pattern': pattern,
        'price': None,
        'description': '',
        'path': image_path
    }

def build_feature_database_subimages(image_folder, model, transform, device):
    """
    构建子图特征数据库：
    只有all文件夹中的图片生成21个子图，其他文件夹中的图片只使用原图
    包含DINOv2深度特征 + 颜色直方图 + Gabor纹理特征
    支持扫描子文件夹并提取属性信息
    """
    supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    original_image_paths = []

    # 递归扫描所有子文件夹
    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            if filename.lower().endswith(supported_ext):
                original_image_paths.append(os.path.join(root, filename))

    if len(original_image_paths) == 0:
        print(f"错误：{image_folder} 中没有找到图片！")
        return None, []

    # 初始化Gabor纹理滤波器
    print("初始化Gabor纹理滤波器...")
    gabor_kernels = create_gabor_kernels(ksize=21, num_theta=8, num_sigma=3)

    features_list = []
    color_features_list = []
    texture_features_list = []
    subimage_info_list = []
    attributes_list = []

    total_subimages = 0

    for i, original_path in enumerate(original_image_paths):
        try:
            print(f"  [{i+1}/{len(original_image_paths)}] 处理: {os.path.basename(original_path)}")
            
            img_cv = cv2.imdecode(np.fromfile(original_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img_cv is None:
                print(f"    无法读取图片: {original_path}")
                continue

            # 提取属性信息
            attributes = extract_attributes_from_path(original_path)

            # 检查是否在all文件夹中
            dir_name = os.path.basename(os.path.dirname(original_path))
            if dir_name == 'all':
                # all文件夹中的图片生成21个子图
                regions_info = get_multiscale_regions_with_info(img_cv, original_path)
                total_subimages += len(regions_info)
            else:
                # 其他文件夹中的图片只使用原图
                h, w = img_cv.shape[:2]
                regions_info = [{
                    'image': img_cv,
                    'crop_type': 'full',
                    'coords': (0, 0, w, h),
                    'original_path': original_path,
                    'img_size': (w, h)
                }]
                total_subimages += 1

            batch_tensors = []
            batch_info = []
            batch_regions = []
            
            for region_info in regions_info:
                pil_img = Image.fromarray(cv2.cvtColor(region_info['image'], cv2.COLOR_BGR2RGB))
                tensor = transform(pil_img)
                batch_tensors.append(tensor)
                batch_info.append({
                    'original_path': region_info['original_path'],
                    'crop_type': region_info['crop_type'],
                    'coords': region_info['coords'],
                    'img_size': region_info['img_size'],
                    'attributes': attributes
                })
                batch_regions.append(region_info['image'])

            # 批量提取DINOv2特征
            batch_tensor = torch.stack(batch_tensors, dim=0).to(device)
            with torch.no_grad():
                batch_features = model(batch_tensor)
            batch_features = F.normalize(batch_features, p=2, dim=1)

            # 提取颜色直方图和纹理特征
            for j, region_img in enumerate(batch_regions):
                color_feat = extract_color_histogram(region_img, bins=32)
                texture_feat = extract_gabor_texture(region_img, gabor_kernels)
                color_features_list.append(color_feat)
                texture_features_list.append(texture_feat)
                attributes_list.append(attributes)

            # 保存DINOv2特征和信息
            for j, (feature, info) in enumerate(zip(batch_features, batch_info)):
                features_list.append(feature.unsqueeze(0))
                subimage_info_list.append(info)

        except Exception as e:
            print(f"    处理失败 {os.path.basename(original_path)}: {e}")

    if not features_list:
        print("错误：没有成功提取任何特征！")
        return None, []

    # 合并所有特征
    feature_matrix = torch.cat(features_list, dim=0)
    color_feature_matrix = np.array(color_features_list, dtype=np.float32)
    texture_feature_matrix = np.array(texture_features_list, dtype=np.float32)
    
    print(f"特征数据库构建完成！共 {total_subimages} 个子图")
    print(f"  - DINOv2特征维度: {feature_matrix.shape}")
    print(f"  - 颜色直方图维度: {color_feature_matrix.shape}")
    print(f"  - Gabor纹理维度: {texture_feature_matrix.shape}")
    
    return feature_matrix, subimage_info_list, color_feature_matrix, texture_feature_matrix



def parse_text_query(query):
    """
    解析文字查询
    新逻辑：搜索框输入的内容会与所有商品信息字段进行文字匹配度计算
    价格区间单独处理（从price_range参数获取）
    """
    if not query:
        return {'query_text': '', 'price_range': None}
    
    query = query.strip()
    
    # 提取价格区间（如果搜索框中包含）
    price_range = None
    match = re.search(r'(\d+(?:\.\d+)?)\s*[-~]\s*(\d+(?:\.\d+)?)\s*元?', query)
    if match:
        min_price = float(match.group(1))
        max_price = float(match.group(2))
        price_range = (min_price, max_price)
        # 从查询中移除价格部分，剩下的作为文字查询
        query = query[:match.start()] + query[match.end():]
        query = query.strip()
    
    return {
        'query_text': query,
        'price_range': price_range
    }

def filter_by_attributes(results, filters):
    """
    根据属性过滤搜索结果
    新逻辑：材质、工艺、花纹使用文本输入匹配，价格区间从商品描述中提取
    """
    print(f"DEBUG: filter_by_attributes called with {len(results)} results")
    print(f"DEBUG: filters = {filters}")
    
    filtered_results = []
    
    for result in results:
        original_path = result['original_path']
        filename = os.path.basename(original_path)
        
        # 从商品信息数据库获取信息
        if filename in product_info_db:
            info = product_info_db[filename]
        else:
            info = {
                'material': '',
                'craft': '',
                'pattern': '',
                'description': ''
            }
        
        # 材质筛选
        if filters.get('material_filter'):
            material_val = info.get('material', '')
            if not material_val or filters['material_filter'] not in material_val:
                continue
        
        # 工艺筛选
        if filters.get('craft_filter'):
            craft_val = info.get('craft', '')
            if not craft_val or filters['craft_filter'] not in craft_val:
                continue
        
        # 花纹筛选
        if filters.get('pattern_filter'):
            pattern_val = info.get('pattern', '')
            if not pattern_val or filters['pattern_filter'] not in pattern_val:
                continue
        
        # 价格区间过滤（从商品描述中提取价格）
        if filters.get('price_range'):
            min_price, max_price = filters['price_range']
            price = extract_price_from_description(info.get('description', ''))
            if price is None or price < min_price or price > max_price:
                continue
        
        filtered_results.append(result)
    
    print(f"DEBUG: filtered to {len(filtered_results)} results")
    return filtered_results

# ============================================================
# 6. 检索函数（原图模式，基于子图匹配 - 新策略）
# ============================================================
def search_similar_images(query_path, feature_matrix, image_paths, model, transform, device, 
                          color_feature_matrix=None, texture_feature_matrix=None, top_k=5, 
                          dino_weight=0.6, color_weight=0.2, texture_weight=0.2):
    """
    在子图数据库中检索，新策略：
    1. 输入图片裁剪为5张（原图 + 2×2的4张）
    2. 与每张原图的21个子图分别比对
    3. 取每张原图的最高匹配度
    4. 返回匹配度最高的前n张
    融合DINOv2特征 + 颜色直方图 + Gabor纹理特征
    """
    img_cv = cv2.imdecode(np.fromfile(query_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_cv is None:
        raise ValueError(f"无法读取查询图片: {query_path}")
    
    h, w = img_cv.shape[:2]
    query_regions = []
    
    query_regions.append({
        'image': img_cv,
        'crop_type': 'full'
    })
    
    for r in range(2):
        for c in range(2):
            y1 = r * h // 2
            y2 = h if r == 1 else (r + 1) * h // 2
            x1 = c * w // 2
            x2 = w if c == 1 else (c + 1) * w // 2
            query_regions.append({
                'image': img_cv[y1:y2, x1:x2],
                'crop_type': f'2x2_{r}_{c}'
            })
    
    # 提取DINOv2特征
    query_features = []
    for region in query_regions:
        pil_img = Image.fromarray(cv2.cvtColor(region['image'], cv2.COLOR_BGR2RGB))
        query_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feature = model(query_tensor)
        feature = F.normalize(feature, p=2, dim=1)
        query_features.append(feature)
    
    # 计算DINOv2相似度
    feature_matrix = feature_matrix.to(device)
    all_similarities = []
    
    for query_feature in query_features:
        similarity = torch.mm(query_feature, feature_matrix.t()).squeeze(0)
        all_similarities.append(similarity)
    
    max_similarities = torch.max(torch.stack(all_similarities), dim=0).values

    # 颜色和纹理特征匹配
    gabor_kernels = create_gabor_kernels(ksize=21, num_theta=8, num_sigma=3)
    
    query_color_features = []
    query_texture_features = []
    
    for region in query_regions:
        color_feat = extract_color_histogram(region['image'], bins=32)
        texture_feat = extract_gabor_texture(region['image'], gabor_kernels)
        query_color_features.append(color_feat)
        query_texture_features.append(texture_feat)
    
    query_color_features = np.array(query_color_features, dtype=np.float32)
    query_texture_features = np.array(query_texture_features, dtype=np.float32)
    
    # 颜色相似度计算
    if color_feature_matrix is not None:
        color_all_similarities = []
        for q_color in query_color_features:
            q_color_norm = q_color / (np.linalg.norm(q_color) + 1e-7)
            similarities = np.dot(color_feature_matrix, q_color_norm)
            color_all_similarities.append(similarities)
        max_color_similarities = np.max(np.stack(color_all_similarities), axis=0)
    else:
        max_color_similarities = np.zeros(len(image_paths))
    
    # 纹理相似度计算
    if texture_feature_matrix is not None:
        texture_all_similarities = []
        for q_texture in query_texture_features:
            q_texture_norm = q_texture / (np.linalg.norm(q_texture) + 1e-7)
            similarities = np.dot(texture_feature_matrix, q_texture_norm)
            texture_all_similarities.append(similarities)
        max_texture_similarities = np.max(np.stack(texture_all_similarities), axis=0)
    else:
        max_texture_similarities = np.zeros(len(image_paths))
    
    # 特征融合权重
    weight_dino = dino_weight
    weight_color = color_weight
    weight_texture = texture_weight

    # 按原图分组，取每张图的最高匹配度
    image_scores = {}
    for idx, info in enumerate(image_paths):
        original_path = info['original_path']
        
        dino_score = max_similarities[idx].item()
        color_score = max_color_similarities[idx]
        texture_score = max_texture_similarities[idx]
        
        fused_score = (weight_dino * dino_score + 
                      weight_color * color_score + 
                      weight_texture * texture_score)
        
        if original_path not in image_scores or fused_score > image_scores[original_path]:
            image_scores[original_path] = fused_score

    sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = sorted_images[:top_k]

    results = []
    for rank, (original_path, score) in enumerate(top_results):
        results.append({
            'rank': rank + 1,
            'original_path': original_path,
            'score': float(score)
        })

    return results

def search_similar_images_original(query_path, feature_matrix_original, image_paths_original, model, transform, device, top_k=5):
    """仅使用DINOv2特征的搜索函数（整体模式）"""
    img_cv = cv2.imdecode(np.fromfile(query_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_cv is None:
        raise ValueError(f"无法读取查询图片: {query_path}")
    
    h, w = img_cv.shape[:2]
    query_regions = []
    
    query_regions.append({
        'image': img_cv,
        'crop_type': 'full'
    })
    
    for r in range(2):
        for c in range(2):
            y1 = r * h // 2
            y2 = h if r == 1 else (r + 1) * h // 2
            x1 = c * w // 2
            x2 = w if c == 1 else (c + 1) * w // 2
            query_regions.append({
                'image': img_cv[y1:y2, x1:x2],
                'crop_type': f'2x2_{r}_{c}'
            })
    
    query_features = []
    for region in query_regions:
        pil_img = Image.fromarray(cv2.cvtColor(region['image'], cv2.COLOR_BGR2RGB))
        query_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feature = model(query_tensor)
        feature = F.normalize(feature, p=2, dim=1)
        query_features.append(feature)
    
    feature_matrix_original = feature_matrix_original.to(device)
    all_similarities = []
    
    for query_feature in query_features:
        similarity = torch.mm(query_feature, feature_matrix_original.t()).squeeze(0)
        all_similarities.append(similarity)
    
    max_similarities = torch.max(torch.stack(all_similarities), dim=0).values

    image_scores = {}
    for idx, info in enumerate(image_paths_original):
        original_path = info['original_path']
        score = max_similarities[idx].item()
        
        if original_path not in image_scores or score > image_scores[original_path]:
            image_scores[original_path] = score

    sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
    top_results = sorted_images[:top_k]

    results = []
    for rank, (original_path, score) in enumerate(top_results):
        results.append({
            'rank': rank + 1,
            'original_path': original_path,
            'score': float(score)
        })

    return results

# ============================================================
# 6. 提取子图并转为Base64
# ============================================================
def extract_subimage_to_base64(original_path, coords):
    """从原图中提取指定区域的子图并转为Base64"""
    try:
        img_cv = cv2.imdecode(np.fromfile(original_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None:
            return None
        
        x1, y1, x2, y2 = coords
        sub_img = img_cv[y1:y2, x1:x2]
        
        # 转换为RGB
        sub_img_rgb = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(sub_img_rgb)
        
        # 转为Base64
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"提取子图失败: {e}")
        return None

def image_to_base64(image_path):
    """将整张图片转为Base64"""
    try:
        with Image.open(image_path) as img:
            # 转换为RGB模式（处理RGBA等模式）
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        print(f"无法转换图片 {image_path}: {e}")
        # 返回一个默认的占位图片
        return None

# ============================================================
# 认证相关路由
# ============================================================
@app.route('/auth/register', methods=['POST'])
def auth_register():
    """用户注册"""
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    display_name = data.get('display_name', '').strip() or username
    
    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400
    
    if len(username) < 2 or len(username) > 20:
        return jsonify({'error': '用户名长度需在2-20个字符之间'}), 400
    
    if len(password) < 4:
        return jsonify({'error': '密码长度不能少于4位'}), 400
    
    if username in users_db:
        return jsonify({'error': '用户名已存在'}), 400
    
    # 创建新用户 - 默认权限
    users_db[username] = {
        'password_hash': hash_password(password),
        'role': 'user',
        'max_results': 10,         # 默认最多查看10张搜索结果
        'can_search': True,
        'can_view_products': True,
        'can_edit_products': False,
        'can_manage_records': False,
        'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'display_name': display_name
    }
    save_users()
    
    # 自动登录
    token = secrets.token_hex(32)
    sessions_db[token] = {
        'username': username,
        'login_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    resp = jsonify({
        'success': True,
        'message': '注册成功',
        'token': token,
        'user': {
            'username': username,
            'role': 'user',
            'display_name': display_name,
            'max_results': 10,
            'can_search': True,
            'can_view_products': True,
            'can_edit_products': False,
            'can_manage_records': False
        }
    })
    resp.set_cookie('auth_token', token, max_age=86400*7, httponly=False, samesite='Lax')
    return resp

@app.route('/auth/login', methods=['POST'])
def auth_login():
    """用户登录"""
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({'error': '用户名和密码不能为空'}), 400
    
    if username not in users_db:
        return jsonify({'error': '用户名或密码错误'}), 401
    
    user = users_db[username]
    if user['password_hash'] != hash_password(password):
        return jsonify({'error': '用户名或密码错误'}), 401
    
    # 生成 token
    token = secrets.token_hex(32)
    sessions_db[token] = {
        'username': username,
        'login_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    resp = jsonify({
        'success': True,
        'message': '登录成功',
        'token': token,
        'user': {
            'username': username,
            'role': user.get('role', 'user'),
            'display_name': user.get('display_name', username),
            'max_results': user.get('max_results', 10),
            'can_search': user.get('can_search', True),
            'can_view_products': user.get('can_view_products', True),
            'can_edit_products': user.get('can_edit_products', False),
            'can_manage_records': user.get('can_manage_records', False)
        }
    })
    resp.set_cookie('auth_token', token, max_age=86400*7, httponly=False, samesite='Lax')
    return resp

@app.route('/auth/logout', methods=['POST'])
def auth_logout():
    """用户登出"""
    token = request.headers.get('X-Auth-Token', '') or request.cookies.get('auth_token', '')
    if token in sessions_db:
        del sessions_db[token]
    resp = jsonify({'success': True, 'message': '已退出登录'})
    resp.delete_cookie('auth_token')
    return resp

@app.route('/auth/me', methods=['GET'])
def auth_me():
    """获取当前登录用户信息"""
    username, user = get_current_user()
    if not username:
        return jsonify({'logged_in': False}), 200
    
    return jsonify({
        'logged_in': True,
        'user': {
            'username': username,
            'role': user.get('role', 'user'),
            'display_name': user.get('display_name', username),
            'max_results': user.get('max_results', 10),
            'can_search': user.get('can_search', True),
            'can_view_products': user.get('can_view_products', True),
            'can_edit_products': user.get('can_edit_products', False),
            'can_manage_records': user.get('can_manage_records', False)
        }
    })

@app.route('/auth/change_password', methods=['POST'])
@login_required
def auth_change_password():
    """修改密码"""
    data = request.json
    old_password = data.get('old_password', '').strip()
    new_password = data.get('new_password', '').strip()
    
    if not old_password or not new_password:
        return jsonify({'error': '请填写完整信息'}), 400
    
    if len(new_password) < 4:
        return jsonify({'error': '新密码长度不能少于4位'}), 400
    
    user = users_db[request.current_user]
    if user['password_hash'] != hash_password(old_password):
        return jsonify({'error': '原密码错误'}), 400
    
    user['password_hash'] = hash_password(new_password)
    save_users()
    return jsonify({'success': True, 'message': '密码修改成功'})

# ============================================================
# 管理员路由
# ============================================================
@app.route('/admin/users', methods=['GET'])
@admin_required
def admin_list_users():
    """获取所有用户列表"""
    user_list = []
    for username, info in users_db.items():
        user_list.append({
            'username': username,
            'role': info.get('role', 'user'),
            'display_name': info.get('display_name', username),
            'max_results': info.get('max_results', 10),
            'can_search': info.get('can_search', True),
            'can_view_products': info.get('can_view_products', True),
            'can_edit_products': info.get('can_edit_products', False),
            'can_manage_records': info.get('can_manage_records', False),
            'created_time': info.get('created_time', '')
        })
    return jsonify({'users': user_list})

@app.route('/admin/user/update', methods=['POST'])
@admin_required
def admin_update_user():
    """管理员更新用户权限"""
    data = request.json
    target_username = data.get('username')
    
    if not target_username or target_username not in users_db:
        return jsonify({'error': '用户不存在'}), 404
    
    # 不允许修改admin自己的角色
    if target_username == 'admin' and data.get('role') and data.get('role') != 'admin':
        return jsonify({'error': '不能修改超级管理员角色'}), 400
    
    user = users_db[target_username]
    
    # 更新可修改的字段
    if 'max_results' in data:
        user['max_results'] = max(1, int(data['max_results']))
    if 'can_search' in data:
        user['can_search'] = bool(data['can_search'])
    if 'can_view_products' in data:
        user['can_view_products'] = bool(data['can_view_products'])
    if 'can_edit_products' in data:
        user['can_edit_products'] = bool(data['can_edit_products'])
    if 'can_manage_records' in data:
        user['can_manage_records'] = bool(data['can_manage_records'])
    if 'role' in data and data['role'] in ('user', 'admin'):
        user['role'] = data['role']
    if 'display_name' in data:
        user['display_name'] = data['display_name'].strip()
    
    save_users()
    return jsonify({'success': True, 'message': f'用户 {target_username} 权限已更新'})

@app.route('/admin/user/delete', methods=['POST'])
@admin_required
def admin_delete_user():
    """管理员删除用户"""
    data = request.json
    target_username = data.get('username')
    
    if not target_username:
        return jsonify({'error': '用户名不能为空'}), 400
    
    if target_username == 'admin':
        return jsonify({'error': '不能删除超级管理员'}), 400
    
    if target_username not in users_db:
        return jsonify({'error': '用户不存在'}), 404
    
    del users_db[target_username]
    
    # 清除该用户的所有session
    tokens_to_remove = [t for t, s in sessions_db.items() if s['username'] == target_username]
    for t in tokens_to_remove:
        del sessions_db[t]
    
    save_users()
    return jsonify({'success': True, 'message': f'用户 {target_username} 已删除'})

@app.route('/admin/user/reset_password', methods=['POST'])
@admin_required
def admin_reset_password():
    """管理员重置用户密码"""
    data = request.json
    target_username = data.get('username')
    new_password = data.get('new_password', '').strip()
    
    if not target_username or target_username not in users_db:
        return jsonify({'error': '用户不存在'}), 404
    
    if not new_password or len(new_password) < 4:
        return jsonify({'error': '新密码长度不能少于4位'}), 400
    
    users_db[target_username]['password_hash'] = hash_password(new_password)
    save_users()
    return jsonify({'success': True, 'message': f'用户 {target_username} 密码已重置'})

# ============================================================
# Flask 路由
# ============================================================
@app.route('/')
def index():
    return render_template('image_detection.html')

@app.route('/search', methods=['POST'])
@login_required
def search():
    if not system_ready:
        return jsonify({'error': '系统尚未初始化完成，请稍后再试'}), 503

    # 权限检查
    user_info = request.current_user_info
    if not user_info.get('can_search', True):
        return jsonify({'error': '您没有搜索权限，请联系管理员'}), 403

    # 获取用户可查看的最大结果数
    user_max_results = user_info.get('max_results', 10)

    file_count = int(request.form.get('file_count', 1))
    text_query = request.form.get('text_query', '')
    material_filter = request.form.get('material_filter', '').strip()
    craft_filter = request.form.get('craft_filter', '').strip()
    pattern_filter = request.form.get('pattern_filter', '').strip()
    price_range = request.form.get('price_range', '').strip()
    
    # 解析文字查询中的价格区间
    parsed_query = parse_text_query(text_query)
    
    # 合并筛选条件
    filters = {
        'material_filter': material_filter,
        'craft_filter': craft_filter,
        'pattern_filter': pattern_filter,
        'price_range': None
    }
    
    # 解析价格区间（优先使用专门的价格区间输入框）
    if price_range:
        try:
            parts = re.split(r'[-~]', price_range)
            if len(parts) == 2:
                min_price = float(parts[0].strip())
                max_price = float(parts[1].strip())
                filters['price_range'] = (min_price, max_price)
        except:
            pass
    elif parsed_query.get('price_range'):
        filters['price_range'] = parsed_query.get('price_range')
    
    # 获取纯文字查询（去除价格部分）
    query_text = parsed_query.get('query_text', '')
    
    try:
        filepaths = []
        weights = []
        
        for i in range(file_count):
            file_key = f'file{i}'
            weight_key = f'weight{i}'
            
            if file_key not in request.files:
                continue
            file = request.files[file_key]
            if file.filename == '':
                continue
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, f'query_{i}_{filename}')
            file.save(filepath)
            filepaths.append(filepath)
            
            weight = float(request.form.get(weight_key, 100))
            weights.append(weight)

        has_filters = material_filter or craft_filter or pattern_filter or filters['price_range']
        if not filepaths and not text_query and not has_filters:
            return jsonify({'error': '没有上传有效的图片文件、文字查询或筛选条件'}), 400

        top_k = int(request.form.get('top_k', user_max_results))
        search_mode = request.form.get('mode', 'fusion')
        
        dino_weight = float(request.form.get('dino_weight', 60)) / 100.0
        color_weight = float(request.form.get('color_weight', 20)) / 100.0
        texture_weight = float(request.form.get('texture_weight', 20)) / 100.0
        
        total_weight = dino_weight + color_weight + texture_weight
        if total_weight > 0:
            dino_weight /= total_weight
            color_weight /= total_weight
            texture_weight /= total_weight
        else:
            dino_weight, color_weight, texture_weight = 0.6, 0.2, 0.2
        
        if len(weights) > 0:
            total_image_weight = sum(weights)
            if total_image_weight > 0:
                weights = [w / total_image_weight for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
        
        print(f"\n开始搜索，模式: {search_mode}，返回 {top_k} 个最相似的原图...")
        print(f"上传图片数量: {len(filepaths)}")
        print(f"文字查询: {text_query}")
        print(f"筛选条件: 材质={material_filter}, 工艺={craft_filter}, 花纹={pattern_filter}, 价格区间={filters['price_range']}")
        start_time = time.time()

        # 获取所有唯一的原图路径
        all_images_info = {}
        for info in image_paths:
            original_path = info.get('original_path', info)
            if original_path not in all_images_info:
                all_images_info[original_path] = {
                    'score': 0.0,
                    'image_score': 0.0,
                    'text_score': 0.0
                }
        
        # 1. 如果有上传图片，计算图片相似度
        if filepaths:
            for filepath, weight in zip(filepaths, weights):
                if search_mode == 'original':
                    results = search_similar_images_original(filepath, feature_matrix_original, image_paths_original, model, transform, device, len(all_images_info))
                else:
                    results = search_similar_images(filepath, feature_matrix, image_paths, model, transform, device, 
                                                     color_feature_matrix, texture_feature_matrix, len(all_images_info), 
                                                     dino_weight, color_weight, texture_weight)
                
                for result in results:
                    original_path = result['original_path']
                    score = result['score']
                    if original_path in all_images_info:
                        all_images_info[original_path]['image_score'] += score * weight
        
        # 2. 如果有文字查询，计算文字匹配度
        if query_text:
            for original_path in all_images_info:
                filename = os.path.basename(original_path)
                if filename in product_info_db:
                    info = product_info_db[filename]
                else:
                    info = extract_attributes_from_path(original_path)
                
                t_score = text_match_score(query_text, info)
                all_images_info[original_path]['text_score'] = t_score
        
        # 3. 综合评分
        for original_path, data in all_images_info.items():
            if filepaths and query_text:
                # 同时有图片和文字查询：结合两个分数
                data['score'] = 0.5 * data['image_score'] + 0.5 * data['text_score']
            elif filepaths:
                data['score'] = data['image_score']
            elif query_text:
                data['score'] = data['text_score']
            else:
                data['score'] = 0.5  # 没有图片也没有文字，纯筛选模式
        
        # 4. 构建结果列表
        avg_results = []
        for original_path, data in all_images_info.items():
            avg_results.append({
                'original_path': original_path,
                'score': data['score']
            })
        avg_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 5. 归一化分数
        if avg_results:
            max_score = max(result['score'] for result in avg_results)
            min_score = min(result['score'] for result in avg_results)
            if max_score > min_score:
                for result in avg_results:
                    result['score'] = (result['score'] - min_score) / (max_score - min_score)
            else:
                for result in avg_results:
                    result['score'] = 0.5
        
        # 6. 应用属性筛选（材质、工艺、花纹、价格区间）
        if has_filters:
            avg_results = filter_by_attributes(avg_results, filters)
        
        # 7. 应用用户权限限制（管理员不限制）
        effective_limit = min(top_k, user_max_results) if user_info.get('role') != 'admin' else top_k
        
        final_results = avg_results[:effective_limit]
        for rank, result in enumerate(final_results):
            result['rank'] = rank + 1

        elapsed_time = time.time() - start_time
        print(f"搜索完成，耗时: {elapsed_time:.2f} 秒 (用户: {request.current_user}, 限制: {user_max_results})")

        # 8. 分页：将完整结果存入缓存，只返回第一页
        search_id = str(uuid.uuid4())
        per_page = int(request.form.get('per_page', 10))
        per_page = max(1, min(per_page, 100))

        # 清理过期缓存
        now = time.time()
        expired_keys = [k for k, v in search_results_cache.items() if now - v['time'] > SEARCH_CACHE_MAX_AGE]
        for k in expired_keys:
            del search_results_cache[k]

        # 存入缓存（只存路径和分数，不存base64图片）
        search_results_cache[search_id] = {
            'results': final_results,
            'time': now,
            'mode': search_mode,
            'text_query': text_query,
            'total_subimages': len(image_paths) if search_mode != 'original' else len(image_paths_original),
            'search_time': elapsed_time,
            'user_max_results': user_max_results,
            'query_image_path': filepaths[0] if filepaths else None
        }

        # 计算分页信息
        total_results = len(final_results)
        total_pages = max(1, (total_results + per_page - 1) // per_page)
        page = 1
        page_start = 0
        page_end = min(per_page, total_results)
        page_results = final_results[page_start:page_end]

        response_results = []
        for result in page_results:
            image_base64 = image_to_base64(result['original_path'])
            attributes = extract_attributes_from_path(result['original_path'])
            response_results.append({
                'rank': result['rank'],
                'original_filename': os.path.basename(result['original_path']),
                'score': result['score'],
                'image': image_base64,
                'attributes': attributes
            })

        query_img_base64 = image_to_base64(filepaths[0]) if filepaths else None

        return jsonify({
            'query_image': query_img_base64,
            'results': response_results,
            'search_time': elapsed_time,
            'total_subimages': len(image_paths) if search_mode != 'original' else len(image_paths_original),
            'mode': search_mode,
            'text_query': text_query,
            'user_max_results': user_max_results,
            'search_id': search_id,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_results': total_results,
                'total_pages': total_pages
            }
        })

    except Exception as e:
        print(f"搜索出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    finally:
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)

@app.route('/search/page', methods=['POST'])
@login_required
def search_page():
    """分页加载搜索结果（从缓存中取出指定页的结果，按需生成base64图片）"""
    data = request.json
    search_id = data.get('search_id', '')
    page = int(data.get('page', 1))
    per_page = int(data.get('per_page', 10))
    per_page = max(1, min(per_page, 100))

    if search_id not in search_results_cache:
        return jsonify({'error': '搜索结果已过期，请重新搜索'}), 404

    cache = search_results_cache[search_id]
    all_results = cache['results']
    total_results = len(all_results)
    total_pages = max(1, (total_results + per_page - 1) // per_page)
    page = max(1, min(page, total_pages))

    page_start = (page - 1) * per_page
    page_end = min(page_start + per_page, total_results)
    page_results = all_results[page_start:page_end]

    response_results = []
    for result in page_results:
        image_base64 = image_to_base64(result['original_path'])
        attributes = extract_attributes_from_path(result['original_path'])
        response_results.append({
            'rank': result['rank'],
            'original_filename': os.path.basename(result['original_path']),
            'score': result['score'],
            'image': image_base64,
            'attributes': attributes
        })

    query_img_base64 = None
    if cache.get('query_image_path'):
        query_img_base64 = image_to_base64(cache['query_image_path'])

    return jsonify({
        'query_image': query_img_base64,
        'results': response_results,
        'search_time': cache.get('search_time', 0),
        'total_subimages': cache.get('total_subimages', 0),
        'mode': cache.get('mode', ''),
        'text_query': cache.get('text_query', ''),
        'user_max_results': cache.get('user_max_results', 10),
        'search_id': search_id,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total_results': total_results,
            'total_pages': total_pages
        }
    })

@app.route('/products', methods=['GET'])
@login_required
def get_products():
    """获取所有商品列表（不含图片数据，图片通过/product_image/接口按需加载）"""
    user_info = request.current_user_info
    if not user_info.get('can_view_products', True) and user_info.get('role') != 'admin':
        return jsonify({'error': '您没有查看商品信息的权限'}), 403
    try:
        products = []
        for filename, info in product_info_db.items():
            products.append({
                'name': filename,
                'material': info.get('material', ''),
                'craft': info.get('craft', ''),
                'pattern': info.get('pattern', ''),
                'description': info.get('description', ''),
                'order_count': len(info.get('order_records', [])),
                'recommend_count': len(info.get('recommend_records', []))
            })
        
        return jsonify({'products': products})
    except Exception as e:
        print(f"获取商品列表出错: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/product_image/<path:filename>')
@login_required
def get_product_image(filename):
    """按需提供单个商品图片（直接返回图片文件）"""
    try:
        if filename in product_info_db:
            filepath = product_info_db[filename].get('path', '')
            if filepath and os.path.exists(filepath):
                # 生成缩略图以减少传输量
                img = Image.open(filepath)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                # 限制最大尺寸为400px
                resample = getattr(Image, 'LANCZOS', getattr(Image, 'ANTIALIAS', None))
                if resample:
                    img.thumbnail((400, 400), resample)
                else:
                    img.thumbnail((400, 400))
                buf = BytesIO()
                img.save(buf, format='JPEG', quality=75)
                buf.seek(0)
                return send_file(buf, mimetype='image/jpeg')
    except Exception as e:
        print(f"生成缩略图失败 {filename}: {e}")
    
    # 返回1x1像素作为fallback
    buf = BytesIO()
    fallback = Image.new('RGB', (1, 1), (245, 245, 247))
    fallback.save(buf, format='JPEG')
    buf.seek(0)
    return send_file(buf, mimetype='image/jpeg')

@app.route('/product/<path:filename>', methods=['GET'])
@login_required
def get_product(filename):
    """获取单个商品详细信息"""
    user_info = request.current_user_info
    if not user_info.get('can_view_products', True) and user_info.get('role') != 'admin':
        return jsonify({'error': '您没有查看商品信息的权限'}), 403
    try:
        if filename not in product_info_db:
            return jsonify({'error': '商品不存在'}), 404
        
        info = product_info_db[filename]
        
        return jsonify({
            'name': filename,
            'material': info.get('material', ''),
            'craft': info.get('craft', ''),
            'pattern': info.get('pattern', ''),
            'description': info.get('description', ''),
            'order_records': info.get('order_records', []),
            'recommend_records': info.get('recommend_records', [])
        })
    except Exception as e:
        print(f"获取商品详情出错: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/product/update', methods=['POST'])
@login_required
def update_product():
    """更新商品信息"""
    user_info = request.current_user_info
    if not user_info.get('can_edit_products', False) and user_info.get('role') != 'admin':
        return jsonify({'error': '您没有编辑商品信息的权限'}), 403
    
    data = request.json
    filename = data.get('name')
    
    if not filename or filename not in product_info_db:
        return jsonify({'error': '商品不存在'}), 404
    
    # 更新可编辑字段
    product_info_db[filename]['material'] = data.get('material', '')
    product_info_db[filename]['craft'] = data.get('craft', '')
    product_info_db[filename]['pattern'] = data.get('pattern', '')
    product_info_db[filename]['description'] = data.get('description', '')
    
    save_product_info()
    return jsonify({'success': True, 'message': '商品信息已更新'})

@app.route('/product/record/add', methods=['POST'])
@login_required
def add_product_record():
    """添加下单/推荐记录"""
    user_info = request.current_user_info
    if not user_info.get('can_manage_records', False) and user_info.get('role') != 'admin':
        return jsonify({'error': '您没有管理记录的权限'}), 403
    
    data = request.json
    filename = data.get('product_name')
    record_type = data.get('record_type')  # 'order' 或 'recommend'
    
    if not filename or filename not in product_info_db:
        return jsonify({'error': '商品不存在'}), 404
    
    if record_type not in ('order', 'recommend'):
        return jsonify({'error': '无效的记录类型'}), 400
    
    region = data.get('region', '').strip()
    customer = data.get('customer', '').strip()
    person = data.get('person', '').strip()
    
    if not region or not customer or not person:
        return jsonify({'error': '区域、客户、下单/推荐人不能为空'}), 400
    
    info = product_info_db[filename]
    order_records = info.get('order_records', [])
    recommend_records = info.get('recommend_records', [])
    
    # 业务规则：市场唯一性约束
    if record_type == 'recommend':
        # 同一商品，同一市场只能有一个待处理推荐
        pending_in_region = [r for r in recommend_records if r['region'] == region and r['status'] == '待处理']
        if pending_in_region:
            return jsonify({'error': f'该市场({region})已有待处理的推荐记录'}), 400
        
        # 已有下单记录的市场不能再推荐
        ordered_in_region = [r for r in order_records if r['region'] == region and r['status'] != '已取消']
        if ordered_in_region:
            return jsonify({'error': f'该市场({region})已有下单记录，不能再推荐'}), 400
    
    if record_type == 'order':
        # 检查下单约束（同一市场已有未取消的下单记录）
        existing_order = [r for r in order_records if r['region'] == region and r['status'] != '已取消']
        if existing_order:
            return jsonify({'error': f'该市场({region})已有下单记录'}), 400
    
    new_record = {
        'id': str(int(time.time() * 1000)),
        'region': region,
        'customer': customer,
        'person': person,
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': '待处理'
    }
    
    if record_type == 'order':
        if 'order_records' not in info:
            info['order_records'] = []
        info['order_records'].insert(0, new_record)
    else:
        if 'recommend_records' not in info:
            info['recommend_records'] = []
        info['recommend_records'].insert(0, new_record)
    
    save_product_info()
    return jsonify({'success': True, 'message': '记录已添加', 'record': new_record})

@app.route('/product/record/update', methods=['POST'])
@login_required
def update_product_record():
    """更新记录状态（取消/下单/已推荐）"""
    user_info = request.current_user_info
    if not user_info.get('can_manage_records', False) and user_info.get('role') != 'admin':
        return jsonify({'error': '您没有管理记录的权限'}), 403
    
    data = request.json
    filename = data.get('product_name')
    record_id = data.get('record_id')
    record_type = data.get('record_type')  # 'order' 或 'recommend'
    new_status = data.get('new_status')  # '已取消', '已下单', '已推荐'
    
    if not filename or filename not in product_info_db:
        return jsonify({'error': '商品不存在'}), 404
    
    info = product_info_db[filename]
    records_key = 'order_records' if record_type == 'order' else 'recommend_records'
    records = info.get(records_key, [])
    
    for record in records:
        if record['id'] == record_id:
            record['status'] = new_status
            save_product_info()
            return jsonify({'success': True, 'message': '状态已更新'})
    
    return jsonify({'error': '记录不存在'}), 404

@app.route('/status')
def status():
    if system_ready:
        return jsonify({
            'status': 'ready',
            'message': '系统已就绪',
            'total_subimages': len(image_paths),
            'device': str(device),
            'mode': '子图检索模式'
        })
    else:
        return jsonify({
            'status': 'initializing',
            'message': '系统正在初始化中...'
        })

# Flask 全局错误处理：确保所有错误返回 JSON 而非 HTML
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': '接口不存在'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': '服务器内部错误'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    print(f"未捕获异常: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 50)
    print("  万斯图像检索系统")
    print("=" * 50)

    # 先初始化商品信息（不依赖ML模型，可以立即完成）
    print("\n正在初始化商品信息...")
    initialize_product_info()

    # 初始化用户系统
    print("\n正在初始化用户系统...")
    load_users()

    # 启动ML模型初始化线程（耗时较长，异步进行）
    import threading
    init_thread = threading.Thread(target=initialize_system)
    init_thread.daemon = True
    init_thread.start()

    print("\nWeb服务器启动中...")
    print("访问地址: http://localhost:5001")
    print("按 Ctrl+C 停止服务器")
    print("=" * 50 + "\n")

    try:
        app.run(debug=False, host='0.0.0.0', port=5001, use_reloader=False)
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器启动错误: {e}")