import os
import argparse
from icrawler.builtin import GoogleImageCrawler

def scrape_food_images(food_name: str, num_images: int, output_dir: str):
    """
    Scrape food images from Google and save them in a folder named after the food.
    """
    save_dir = os.path.join(output_dir, food_name.replace(" ", "_").lower())
    os.makedirs(save_dir, exist_ok=True)

    google_crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    google_crawler.crawl(keyword=food_name, max_num=num_images)

    # Count actual images in folder
    actual_downloaded = len([
        f for f in os.listdir(save_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Downloaded {actual_downloaded}/{num_images} images of '{food_name}' into {save_dir}")


def load_food_list(food_list_file: str):
    """
    Load food names from a file (one food name per line).
    """
    with open(food_list_file, "r", encoding="utf-8") as f:
        foods = [line.strip() for line in f if line.strip()]
    return foods


def main(args):
    if args.food_list:
        foods = load_food_list(args.food_list)
    else:
        foods = args.foods

    print(f"Scraping {len(foods)} food categories...")

    for food in foods:
        scrape_food_images(food, num_images=args.num_images, output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape food images from Google")
    parser.add_argument("--foods", nargs="+", help="List of food names (inline)")
    parser.add_argument("--food-list", type=str, help="Path to file containing food names (one per line)")
    parser.add_argument("--num-images", type=int, default=50, help="Number of images per food")
    parser.add_argument("--output-dir", type=str, default="datasets/raw/scraped/images", help="Directory to save scraped images")

    args = parser.parse_args()
    main(args)