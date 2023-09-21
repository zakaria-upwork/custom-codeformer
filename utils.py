import cv2

def generate_eye(overlay_image,x,y):
    base_image = cv2.imread('output_eyes_only.png',cv2.IMREAD_UNCHANGED)
    x,y = x+15,y+15
    # Overlay the image without transparency
    height, width, _ = overlay_image.shape
    roi = base_image[y:y+height, x:x+width]
    overlayed = cv2.addWeighted(roi, 0, overlay_image, 1, 0)
    base_image[y:y+height, x:x+width] = overlayed

    # Save the merged image to a file
    cv2.imwrite('output_eyes_only.png', base_image)


def generate_eyes_only(img,image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load the image
    # image = cv2.imread('output.png')

    # Convert the image to grayscale for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect eyes in the image
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    cv2.imwrite('output_eyes_only.png', img)
    for eye in eyes:
        x, y,e_h,e_w=eye
        f_x,f_y,h,w = faces[0]
        # cv2.rectangle(image, (x, y), (x + int(w*0.6), y + e_h), (0, 255, 0), 2)
        cropped_image = image[y+15:y + int(e_h*0.7), x+15:x+int(e_w*0.9)]
        generate_eye(cropped_image,x,y)
