import cv2
import subprocess
import shutil

def download_weights():
    # URL of the file you want to download
    url = "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"

    # Directory where you want to save the downloaded file
    download_directory = "/path/to/download/directory"

    # Directory where you want to move the downloaded file
    destination_directory = "weights/CodeFormer"

    # Use wget to download the file
    try:
        subprocess.run(["wget", url, "-P", download_directory], check=True)
        print("File downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading the file: {e}")

    # Now, move the downloaded file to the destination directory
    try:
        file_name = url.split("/")[-1]  # Extract the file name from the URL
        source_path = f"{download_directory}/{file_name}"
        destination_path = f"{destination_directory}/{file_name}"
        shutil.move(source_path, destination_path)
        print(f"File moved to {destination_path}")
    except Exception as e:
        print(f"Error moving the file: {e}")


def generate_eye(overlay_image,x,y):
    base_image = cv2.imread('output_eyes_only.png',cv2.IMREAD_UNCHANGED)
    x,y = x+10,y+10
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
        cropped_image = image[y+10:y + int(e_h*0.6), x+10:x+e_w]
        generate_eye(cropped_image,x,y)