from pathlib import Path 
from collections import Counter 
import os 
import cv2
from hashlib import sha256 
class DataAnalysis: 
    def __init__(self, root): 
        self.root = root 

        self.classes = os.listdir(root) 
        self.class_counts = {} 
        self.image_paths = []
        # self.labels = [] 
        for class_name in self.classes: 
            class_path = Path(root) / class_name 
            image_count = len(os.listdir(class_path))
            self.class_counts[class_name] = image_count
            for image_name in os.listdir(class_path): 
                image_path = class_path / image_name 
                self.image_paths.append(image_path)

    # So luong anh
    def count_total_images(self): 
        return sum(self.class_counts.values()) 
    # So luong class   
    def count_classes(self): 
        return self.classes, len(self.classes) 
    # So luong anh moi class 
    def count_images_per_class(self): 
        return self.class_counts 
    # So luong anh theo kich thuoc 
    def analyze_image_dimensions(self): 
        sizes = Counter() 
        for image_path in self.image_paths: 
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2] 
                sizes[(h, w)] += 1 
        return len(sizes)
    # So luong anh theo dinh dang 
    def analyze_image_format(self): 
        formats = Counter() 
        for image_name in self.image_paths: 
            ext = os.path.splitext(image_name)[1].lower() 
            formats[ext] += 1 
        return formats 
    # file anh bi hong 
    def find_corrupted_images(self): 
        corrupted_images = [] 
        for image_path in self.image_paths: 
            try: 
                img = cv2.imread(image_path) 
                if img is None: 
                    corrupted_images.append(image_path) 
            except Exception as e: 
                corrupted_images.append(image_path)
        return corrupted_images
    # anh trung lap
    def find_duplicate_images(self): 
        duplicated_images = [] 
        hashes = {} 
        for image_path in self.image_paths: 
            try: 
                img = cv2.imread(image_path) 
                if img is not None: 
                    img_hash = sha256(img.tobytes(),usedforsecurity=True).hexdigest() 
                    if img_hash in hashes: 
                        duplicated_images.append(image_path) 
                    else: 
                        hashes[img_hash] = image_path 
            except Exception as e: 
                continue
        return duplicated_images
def main(): 
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    analizer = DataAnalysis(PROJECT_DIR / "data_collection" / "train")
    print("=" * 50)
    print("SUMMARY OF DATASET") 
    print("=" * 50) 
    print(f"Total images: {analizer.count_total_images()}")
    # print(f"Number of classes: {analizer.count_classes()}")
    classes, num_classes = analizer.count_classes() 
    print(f"Number of classes: {num_classes}, Classes: {classes}")
    print(f"Images per class:")
    images_per_class = analizer.count_images_per_class() 
    for class_name, count in images_per_class.items(): 
        print(f"Class '{class_name}': {count} images") 
    # print(f"Image dimensions:")
    sizes = analizer.analyze_image_dimensions()
    print(f"Number of unique image dimensions: {sizes}")
    # for size, count in sizes.items():
    #     print(f"Size {size}: {count} images")
    print(f"Image formats:")
    formats = analizer.analyze_image_format()
    for fmt, count in formats.items():
        print(f"Format '{fmt}': {count} images")
    print(f"Corrupted images:")
    corrupted_images = analizer.find_corrupted_images()
    if corrupted_images:
        print("List of corrupted images:")
        for img in corrupted_images:
            print(img)
    else:
        print("No corrupted images found.")
    print(f"Duplicate images:")
    duplicated_images = analizer.find_duplicate_images()
    if duplicated_images:
        print("List of duplicate images:")
        for img in duplicated_images:
            print(img)
    else:
        print("No duplicate images found.")
    # def analyze_class_distribution(self): 
    #     pass 
if __name__ == "__main__": 
    main()  









