import os
import os.path
import imutils
import cv2
import glob

image_folder = glob.glob(os.path.join("captchas", "*"))
counts = {}


for (i, base_image) in enumerate(image_folder):
    print("image {}".format(i + 1, ))

    #get solved captcha code
    filename = os.path.basename(base_image)
    captcha_correct_text = os.path.splitext(filename)[0]

    #load image and convert to greyscale
    image = cv2.imread(base_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #add padding
    image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    #convert to black and white
    bandwhite = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #find contours
    contours = cv2.findContours(bandwhite.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0]
    #make array to store bounds for every charachter
    charachter_bounds = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if w / h > 1.30:
            new_width = int(w / 2)
            charachter_bounds.append((x, y, new_width, h))
            charachter_bounds.append((x + new_width, y, new_width, h))
        else:
            charachter_bounds.append((x, y, w, h))

    if len(charachter_bounds) != 4:
        continue

    #sort by x-coordinate
    charachter_bounds = sorted(charachter_bounds, key=lambda x: x[0])

    for bounds, character in zip(charachter_bounds, captcha_correct_text):
        
        x, y, w, h = bounds

        #crop image around letter
        cropped_image = image[y - 2:y + h + 2, x - 2:x + w + 2]

        #get folder
        path = os.path.join("letters", character)

        #if folder doesnt exist create it
        if not os.path.exists(path):
            os.makedirs(path)

        #save image to correct folder
        count = counts.get(character, 1)
        pt = os.path.join(path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(pt, cropped_image)

        counts[character] = count + 1
