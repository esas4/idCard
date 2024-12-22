import random

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

# def main():
    # for i in range(1,51):
    #     year, month, day=random_birthday()
    #     print(f"{year}-{month}-{day}")

# 随机生成号码
def random_number():
    return ''.join(random.choices("0123456789", k=18))

def get_unique_digits(number):
    unique_digits = sorted(set(number))
    return ''.join(unique_digits)

# def main():
    # for i in range(1,2):
    #     number=random_number()
    #     unique_digits = get_unique_digits(number)
    #     print(f"{number} 包含的数字：{unique_digits}")

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

def main():
    for i in range(1,101):
        name=random_name()
        print(name)
        get_sex = lambda :random.choice(['男','女'])
        print(get_sex())
        year, month, day=random_birthday()
        print(f"{year}-{month}-{day}")
        number=random_number()
        # unique_digits = get_unique_digits(number)
        print(number)  
        
main()