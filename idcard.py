import os
import random
from PIL import Image, ImageDraw, ImageFont

# 配置图片文件夹路径
boy_folder = "/home/data/shizh/real_or_fake/idCard/idpics_for_edit/boy"  # 替换为实际的boy文件夹路径
girl_folder = "/home/data/shizh/real_or_fake/idCard/idpics_for_edit/girl"  # 替换为实际的girl文件夹路径
output_folder = "/home/data/shizh/real_or_fake/idCard/3_idcard_pics"  # 替换为保存生成卡片的文件夹路径

# 加载模板图片
template_path = "/home/data/shizh/real_or_fake/idCard/id_example.png"
template_img = Image.open(template_path)

# 随机生成姓名
first_names = [
    "张", "李", "王", "刘", "陈", "杨", "赵", "黄", "周", "吴", 
    "朱", "胡", "郑", "林", "何", "高", "邓", "付", "孙", "马", 
    "丁", "任", "费", "冯", "梁", "唐", "邱", "谢", "蔡", "彭",
    "阮", "魏", "苏", "杜", "李", "段", "蔡", "贾", "盛", "温"
]    

new_first_names = [
    "徐", "罗", "程", "叶", "史", "郭", "宋", "熊", "纪", "舒",
    "屈", "项", "祝", "董", "袁", "穆", "萧", "简", "葛", "顾",
    "孟", "秦", "乔", "邢", "路", "庞", "樊", "栾", "庄", "严",
    "习", "辛", "阎", "连", "翟", "鞠", "缪", "杭", "嵇", "郎"
]

last_names = [
    "伟", "芳", "娜", "秀英", "敏", "静", "丽", "强", "磊", 
    "军", "彬", "梅", "燕", "婷", "宇", "佳", "凯", "雪", 
    "欣", "伟杰", "博", "思", "悦", "晴", "晨", "宁", "琪", 
    "皓", "铭", "华", "灿", "云", "峰", "琪", "欣妍", "思远"
    ]   

new_last_names = [
    "浩", "涵", "洋", "涛", "淼", "淳", "泽", "润", "波", "澜",
    "溪", "洁", "清", "淳", "渊", "淳", "瀚", "鸿", "沛", "沣",
    "澄", "澈", "添", "溶", "涵", "泳", "泰", "泉", "浚", "淳",
    "泽", "洲", "滨", "源", "浩淼", "若水", "润泽", "涛声", "澄泓",
    "伟豪", "芳菲", "娜娜", "秀英", "敏轩", "静娴", "丽娟", "强辉", "磊石",
    "军威", "彬彬", "梅香", "燕飞", "婷婷", "宇航", "佳琪", "凯旋", "雪莲",
    "欣怡", "伟杰", "博远", "思涵", "悦琳", "晴雯", "晨曦", "宁静", "琪瑶",
    "皓轩", "铭瑄", "华丽", "灿阳", "云翔", "峰岳", "琪琪", "欣妍", "思远",
    "浩淼", "涵钰", "洋溢", "涛涛", "淼淼", "淳朴", "泽宇", "润泽", "波澜",
    "澜珊", "溪流", "洁莹", "清澈", "清雅", "淳厚", "渊博", "瀚海", "鸿雁",
    "沛然", "沣沛", "澄澈", "澈净", "添彩", "泰安",
    "泉水", "浚源", "泽润", "洲际", "滨江", "源泉", "浩淼", "若水",
    "海涵", "润泽", "涛声", "澄泓", "嘉诚", "晨浩", "子涵", "雨泽",
    "俊熙", "梓涵"
] 
# return random.choice(first_names) + random.choice(last_names)

# 存储已生成的名字
generated_names = set()

def random_name():
    while True:
        first_name = random.choice(new_first_names)
        last_name = random.choice(new_last_names)
        full_name = first_name + last_name
        
        if full_name not in generated_names:
            generated_names.add(full_name)
            return full_name

# 随机生成生日和号码
def random_birthday():
    year = random.randint(1950, 2020)
    month = random.randint(1, 12)
    if month == 2:
        day = random.randint(1, 28)  # 为简单起见，防止日期溢出，选择1-28日
    elif month == 4 or month == 6 or month == 9 or month == 11:
        day = random.randint(1, 30)
    else:
        day = random.randint(1, 31)
    return year, month, day


def random_number():
    return ''.join(random.choices("0123456789", k=18))

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 加载字体（请根据实际字体文件路径替换）
font_path = "/home/data/shizh/real_or_fake/idCard/SIMHEI.TTF"
font_large = ImageFont.truetype(font_path, 70)
font_medium = ImageFont.truetype(font_path, 70)
font_small = ImageFont.truetype(font_path, 70)

# 处理图片文件
def generate_id_card(image_path, gender):
    # 打开模板并复制
    card = template_img.copy()
    draw = ImageDraw.Draw(card)
    
    # 生成随机信息
    name = random_name()
    year, month, day = random_birthday()
    number = random_number()

    # 定位信息的位置并填写内容
    draw.text((283, 247), name, fill="black", font=font_large)  # 姓名
    draw.text((283, 402), "男" if gender == "男" else "女", fill="black", font=font_large)  # 性别
    draw.text((283, 545), str(year), fill="black", font=font_large)  # 出生年pr
    if month > 9:
        draw.text((581, 545), str(month), fill="black", font=font_large)  # 出生年
    else:
        draw.text((599, 545), str(month), fill="black", font=font_large)
    if day > 9:
        draw.text((750, 545), str(day), fill="black", font=font_large)
    else:
        draw.text((770, 545), str(day), fill="black", font=font_large)  # 出生年
    draw.text((283, 693), number, fill="black", font=font_large)  # 号码

    # 加载并粘贴头像
    profile_img = Image.open(image_path).resize((354, 496))
    card.paste(profile_img, (1181, 256))

    # 保存生成的图片
    output_path = os.path.join(output_folder, f"{name}_{number}.png")
    card.save(output_path)

# 遍历图片文件夹，生成卡片
for gender_folder, gender in [(boy_folder, "男"), (girl_folder, "女")]:
    for filename in os.listdir(gender_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(gender_folder, filename)
            generate_id_card(image_path, gender)
