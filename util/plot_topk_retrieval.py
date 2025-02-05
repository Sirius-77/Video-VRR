import matplotlib.pyplot as plt
from PIL import Image

from img_evaler import eval_img


def plot_topk_retrieval(
        img_path=None,
        topK=3
):
    search_result = eval_img(img_path)
    ref_path = []
    ref_path.append(img_path)
    for i in range(topK):
        ref_path.append(f"../DS/NYCDataset/streetview/ny_persp/{search_result[i]}.jpg")
    # 设置图像数量和排列
    num_images = len(ref_path)
    fig, axs = plt.subplots(1, num_images, figsize=(15, 5))
    titles = ["Query Frame", "Top-1", "Top-2", "Top-3"]
    # 遍历每个图像并绘制
    for i, (url, title) in enumerate(zip(ref_path, titles)):
        img = Image.open(url)

        # 显示图像
        axs[i].imshow(img)
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
    return
