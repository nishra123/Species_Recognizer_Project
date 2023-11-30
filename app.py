import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO model
yolo = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")

# Load COCO names
with open("./coco.names", 'r') as f:
    classes = f.read().splitlines()

def process_image(uploaded_file):
    # Read image from BytesIO object
    content = uploaded_file.getvalue()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)
    output_layers_names = yolo.getUnconnectedOutLayersNames()
    layeroutput = yolo.forward(output_layers_names)

    height, width, _ = img.shape

    boxes = []
    confidences = []
    class_ids = []

    for output in layeroutput:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            if confidence > 0.7:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)

                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    result = []
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = round(confidences[i], 2)
        result.append({'label': label, 'confidence': confidence, 'box': (x, y, x+w, y+h)})

    # Draw boxes on the image
    for item in result:
        x, y, x_end, y_end = item['box']
        cv2.rectangle(img, (x, y), (x_end, y_end), (0, 255, 0), 2)
        cv2.putText(img, f"{item['label']} {item['confidence']}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img, result

def main():
    st.title('Species Recognition App')
    
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption=f'Uploaded Image: {uploaded_file.name}', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Process the image and get the result
        processed_image, result = process_image(uploaded_file)

        # Display the result
        st.write("## Results:")
        if result:
            for item in result:
                st.write(f"- {item['label']} - Confidence: {item['confidence']} - Box: {item['box']}")

            # Display the final image using plt.imshow
            plt.imshow(processed_image)
            plt.axis('off')
            st.pyplot()

        else:
            st.write("No result available")

if __name__ == '__main__':
    main()
