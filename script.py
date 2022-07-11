import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
import shutil
import os
from itertools import groupby
start_time = time.time()

IMAGE_PATH = "polaris2.png"
LANGUAGE_SET_1 = ['en']

class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.image = ""
        self.Text = []

### Performs OCR on a screenshot specified in IMAGE_PATH variable
### Returns the list of words and theirs locations
def Image_To_Text(PATH_TO_IMAGE, SourceLanguage):
    result = []

    try:
        reader = easyocr.Reader([SourceLanguage], gpu=False)
        result = result + reader.readtext(PATH_TO_IMAGE)
        return True, 'OK', result
    except Exception as e:
        print(str(e))
        return False, 'Failed to OCR the image:' + PATH_TO_IMAGE  + str(e), result

    return True, 'OK', result

### Processess image in different ways such as applying blurs, erosion, dilations etc...
def Pre_Process_Image(DirectoryPath):
    try:
        original = cv2.imread(IMAGE_PATH)
        cv2.imwrite(DirectoryPath + '/original.png', original)

        imagem = cv2.bitwise_not(original)
        cv2.imwrite(DirectoryPath + '/bitwise_not.png', imagem)

        imagem2 = cv2.bitwise_not(imagem)
        cv2.imwrite(DirectoryPath + '/bitwise.png', imagem2)

        blur1 = cv2.blur(original,(5,5))
        cv2.imwrite(DirectoryPath + '/blur1.png', blur1)

        GaussianBlur = cv2.GaussianBlur(original, (5, 5), 0)
        cv2.imwrite(DirectoryPath + '/GaussianBlur.png', GaussianBlur)

        medianBlur = cv2.medianBlur(original, 3)
        cv2.imwrite(DirectoryPath + '/medianBlur.png', medianBlur)

        bilateralFilter = cv2.bilateralFilter(original,9,75,75)
        cv2.imwrite(DirectoryPath + '/bilateralFilter.png', bilateralFilter)

        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(original,kernel,iterations = 1)
        cv2.imwrite(DirectoryPath + '/erosion.png', erosion)

        dilation = cv2.dilate(original,kernel,iterations = 1)
        cv2.imwrite(DirectoryPath + '/dilation.png', dilation)

        opening = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(DirectoryPath + '/opening.png', dilation)

        closing = cv2.morphologyEx(original, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(DirectoryPath + '/closing.png', dilation)

        gradient = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel)
        cv2.imwrite(DirectoryPath + '/gradient.png', dilation)

        tophat = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel)
        cv2.imwrite(DirectoryPath + '/tophat.png', dilation)

        blackhat = cv2.morphologyEx(original, cv2.MORPH_BLACKHAT, kernel)
        cv2.imwrite(DirectoryPath + '/blackhat.png', dilation)

        return True, 'OK'
    except Exception as e:
        return False, 'Failed to Pre-process image. ' + str(e)

### Places screenshots of regions text into specified folder
def Find_Contours(SourceDirectoryPath, TargetDirectoryPath):
    Rectangles = []

    names = [
        "original.png",
        "bitwise_not.png",
        "bitwise.png",
        "blur1.png",
        "GaussianBlur.png",
        "medianBlur.png",
        "bilateralFilter.png",
        "erosion.png",
        "dilation.png",
        "opening.png",
        "closing.png",
        "gradient.png",
        "tophat.png",
        "blackhat.png"
    ]

    for name in names:
        try:
            large = cv2.imread(SourceDirectoryPath + "/" + name)
            rgb = None
            crop_source = None

            try:
                xScale = 0
                yScale = 0

                rgb = cv2.pyrDown(large)

                orig_height, orig_width = large.shape[:2]
                new_height, new_width = rgb.shape[:2]
                xScale = orig_height/new_height
                yScale = orig_width/new_width
                crop_source = cv2.imread(SourceDirectoryPath + "/" + name)

                pass
            except Exception as e:
                print(str(e))
                break

            small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
            _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            mask = np.zeros(bw.shape, dtype=np.uint8)

            for idx in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[idx])
                mask[y:y+h, x:x+w] = 0
                cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
                r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

                if r > 0.45 and w > 3 and h > 3:
                    x = int(x * xScale)
                    y = int(y * xScale)
                    w = int(w * xScale)
                    h = int(h * xScale)

                    if orig_height > 300 and orig_width > 300:
                        heightDifference = abs(orig_height - h)
                        widthDifference = abs(orig_width - w)

                        if heightDifference < 10 or widthDifference < 10:
                            continue

                    if(not any(rect.w == w and rect.h == h and rect.y == y and rect.x == x for rect in Rectangles)):
                        Rectangles.append(Rectangle(x, y, w, h))

        except Exception as e:
            return False, 'Failed to Find contours for the image. ' + IMAGE_PATH + str(e), Rectangles

    return True, 'OK', Rectangles

def DeleteDirectory(DirectoryPath):
    try:
        if(os.path.isdir(DirectoryPath)):
            shutil.rmtree(DirectoryPath)

        if(os.path.isdir(DirectoryPath)):
            return False, 'Failed to remove the directory:' +  DirectoryPath

        return True, 'OK'
    except Exception as e:
        return False, 'Failed to remove the directory: ' +  DirectoryPath + '. ' + str(e)

def CreateDirectory(DirectoryPath):
    try:
        os.makedirs(DirectoryPath)
        if(not os.path.isdir(DirectoryPath)):
            return False, 'Failed to create the directory: ' + DirectoryPath  + '.'

        return True, 'OK'
    except Exception as e:
        return False, 'Failed to create the directory:' + DirectoryPath  + str(e)

def CropRectanglesFromOriginal(Rectangles, SourceDirectoryPath, TargetDirectoryPath):
    try:
        original_image_drawn_rectangles = cv2.imread(SourceDirectoryPath + "/" + "original.png")

        original_image = cv2.imread(SourceDirectoryPath + "/" + "original.png")
        for idx in range(len(Rectangles)):
            x = Rectangles[idx].x
            y = Rectangles[idx].y
            w = Rectangles[idx].w
            h = Rectangles[idx].h
            crop_from_original = original_image[y:y+h, x:x+w]
            Rectangles[idx].image = crop_from_original
            cv2.rectangle(original_image_drawn_rectangles, (x, y), ((x+w), (y+h)  ), (255,0,0), 2)

        sorted_letters = sorted(Rectangles, key=lambda Rectangle: str(Rectangle.image))
        grouped = [list(result) for key, result in groupby(sorted_letters, key=lambda Rectangle: str(Rectangle.image))]

        for idx_g in range(len(grouped)):
            cv2.imwrite(TargetDirectoryPath + '/' + str(idx_g + 1) + ".png", grouped[idx_g][0].image)
            cv2.rectangle(original_image_drawn_rectangles, (grouped[idx_g][0].x, grouped[idx_g][0].y), (grouped[idx_g][0].x + grouped[idx_g][0].w, grouped[idx_g][0].h + grouped[idx_g][0].y), (255,0,0), 2)

        cv2.imwrite("output.png", original_image_drawn_rectangles)

        return True, 'OK'
    except Exception as e:
         return False, 'Failed to CropRectanglesFromOriginal' + str(e)



def Process_Rectangles(Rectangles, SourceLanguage):
    try:
        for idx in range(len(Rectangles)):
            Rectangles[idx].Text = Image_To_Text("image_text_regions/" +  str(idx+1) + ".png", SourceLanguage)
        print("len2 is " + str(len(Rectangles)))
        return True, 'OK', Rectangles
    except Exception as e:
         return False, 'Failed to Process_Rectangles' + str(e)

def PrintRectangles(Rectangles):
    try:
        print("print rectangles")
        for idx in range(len(Rectangles)):
            print(str(idx+1) + ":\n" + str(Rectangles[idx].Text) )
        return True, 'OK'
    except Exception as e:
         print(str(e))
         return False, 'Failed to CropRectanglesFromOriginal' + str(e)

if __name__ == "__main__":
    res1, message1 = DeleteDirectory('pre_ocr_processed_images')
    res2, message2 = CreateDirectory('pre_ocr_processed_images')
    res3, message3 = Pre_Process_Image('pre_ocr_processed_images')
    res4, message4 = DeleteDirectory('image_text_regions')
    res5, message5 = CreateDirectory('image_text_regions')
    res6, message6, Rectangles = Find_Contours("pre_ocr_processed_images", "image_text_regions")
    res8, message8 = CropRectanglesFromOriginal(Rectangles, 'pre_ocr_processed_images', "image_text_regions")
    res9, message9, Rectangles23 = Process_Rectangles(Rectangles, "en")
    print("len is " + str(len(Rectangles23)))
    PrintRectangles(Rectangles23)



    print("--- %s seconds ---" % (time.time() - start_time))
