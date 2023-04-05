from PIL import Image

for i in range(1, 28):
    # Read the two images
    try:
        image1 = Image.open('Data/Ordered/%da.jpg' % i)
        # image1.show()
        image2 = Image.open('Data/Ordered/%db.jpg' % i)
        # image2.show()

        # resize, first image
        # image1 = image1.resize((228, 228))
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB', (2 * image1_size[0], image1_size[1]), (250, 250, 250))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1_size[0], 0))
        new_image.save("Data/Merged/merged_image%d.jpg" % i, "JPEG")
        # new_image.show()
    except FileNotFoundError:
        continue



