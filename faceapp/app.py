from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import logging
import uvicorn  # Added uvicorn import

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Scan Service")

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings for stabilization
COSINE_THRESHOLD = 0.6      # Threshold for face similarity check
ROTATION_ANGLES = [-15, 0, 15]  # Rotation angles for augmentation
QUANTIZATION_BINS = 16      # Number of discrete bins for quantization

# Initialize face analyzer
@app.on_event("startup")
async def startup_event():
    global face_analyzer
    logger.info("Initializing face analyzer...")
    face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    logger.info("Face analyzer initialized successfully")

def preprocess_image(image_bytes):
    """Preprocess image before analysis"""
    try:
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Failed to decode image")

        # Check image size
        if img.shape[0] < 100 or img.shape[1] < 100:
            # Scale image if too small
            ratio = 640 / min(img.shape[0], img.shape[1])
            new_size = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
            img = cv2.resize(img, new_size)

        # Normalize brightness and contrast (adaptive histogram equalization)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        return img
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def align_face(img, landmarks):
    """Align face based on key points"""
    try:
        # Use eye positions for alignment
        left_eye = landmarks[0]
        right_eye = landmarks[1]

        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Center of rotation - midpoint between eyes
        center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)

        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)

        # Rotate image
        height, width = img.shape[:2]
        aligned_img = cv2.warpAffine(img, M, (width, height),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)

        return aligned_img
    except Exception as e:
        logger.warning(f"Face alignment failed: {str(e)}. Using original image.")
        return img

def augment_and_extract_embeddings(img, face):
    """Create multiple embeddings with rotations and brightness variations"""
    embeddings = []

    # Align face based on key points
    if hasattr(face, 'kps') and face.kps is not None:
        aligned_img = align_face(img, face.kps)
    else:
        aligned_img = img

    # Extract embedding for aligned image
    crop_size = (112, 112)  # Standard size for ArcFace

    # Crop face with padding for context
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    # Add 20% padding on each side
    w, h = x2 - x1, y2 - y1
    x1 = max(0, x1 - int(w * 0.2))
    y1 = max(0, y1 - int(h * 0.2))
    x2 = min(aligned_img.shape[1], x2 + int(w * 0.2))
    y2 = min(aligned_img.shape[0], y2 + int(h * 0.2))

    face_img = aligned_img[y1:y2, x1:x2]
    if face_img.size == 0:
        # Fallback to original face crop if cropping fails
        face_img = aligned_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]


    # Check if image is empty
    if face_img.size == 0:
        logger.warning("Face crop resulted in empty image, using full image")
        face_img = aligned_img

    # Get base embedding
    base_embedding = face.embedding
    embeddings.append(base_embedding)

    # Apply rotations to account for head tilts
    height, width = face_img.shape[:2]
    center = (width // 2, height // 2)

    for angle in ROTATION_ANGLES:
        if angle == 0:  # Skip 0 angle as it's the base embedding
            continue

        # Rotate image
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(face_img, M, (width, height))

        # Resize for ArcFace model
        rotated = cv2.resize(rotated, crop_size)

        # Convert to RGB for model compatibility
        rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

        # Process rotated image
        temp_img = aligned_img.copy()
        y_offset = (temp_img.shape[0] - height) // 2
        x_offset = (temp_img.shape[1] - width) // 2

        try:
            # Create ROI for insertion
            y1_roi = max(0, y_offset)
            y2_roi = min(temp_img.shape[0], y_offset + height)
            x1_roi = max(0, x_offset)
            x2_roi = min(temp_img.shape[1], x_offset + width)

            # Insert rotated image
            h_roi = y2_roi - y1_roi
            w_roi = x2_roi - x1_roi
            temp_img[y1_roi:y2_roi, x1_roi:x2_roi] = cv2.resize(rotated, (w_roi, h_roi))

            # Analyze rotated image
            temp_faces = face_analyzer.get(temp_img)

            if temp_faces:
                temp_face = max(temp_faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
                rotated_embedding = temp_face.embedding

                # Check rotation doesn't deviate too much
                cosine_similarity = np.dot(base_embedding, rotated_embedding) / (
                    np.linalg.norm(base_embedding) * np.linalg.norm(rotated_embedding))

                if cosine_similarity > COSINE_THRESHOLD:
                    embeddings.append(rotated_embedding)
            else:
                logger.debug(f"No face detected in rotated image with angle {angle}")
        except Exception as e:
            logger.warning(f"Error processing rotated face at angle {angle}: {str(e)}")

    # Add brightness/contrast variations
    try:
        bright_img = cv2.convertScaleAbs(aligned_img, alpha=1.1, beta=10)
        dark_img = cv2.convertScaleAbs(aligned_img, alpha=0.9, beta=-10)

        bright_faces = face_analyzer.get(bright_img)
        dark_faces = face_analyzer.get(dark_img)

        if bright_faces:
            bright_face = max(bright_faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            bright_embedding = bright_face.embedding
            cosine_similarity = np.dot(base_embedding, bright_embedding) / (
                np.linalg.norm(base_embedding) * np.linalg.norm(bright_embedding))
            if cosine_similarity > COSINE_THRESHOLD:
                embeddings.append(bright_embedding)

        if dark_faces:
            dark_face = max(dark_faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))
            dark_embedding = dark_face.embedding
            cosine_similarity = np.dot(base_embedding, dark_embedding) / (
                np.linalg.norm(base_embedding) * np.linalg.norm(dark_embedding))
            if cosine_similarity > COSINE_THRESHOLD:
                embeddings.append(dark_embedding)
    except Exception as e:
        logger.warning(f"Error processing brightness variations: {str(e)}")

    return embeddings

def normalize_embedding(embeddings):
    """Average and normalize embeddings for stable representation"""
    if not embeddings:
        raise ValueError("No valid embeddings to normalize")

    # Compute mean embedding
    mean_embedding = np.mean(embeddings, axis=0)

    # L2 normalization
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm

    # Quantize for stability
    quantized = quantize_embedding(mean_embedding, QUANTIZATION_BINS)

    return quantized

def quantize_embedding(embedding, num_bins):
    """Quantize embedding for stability"""
    min_val = -1.0
    max_val = 1.0
    normalized = (embedding - min_val) / (max_val - min_val)
    quantized = np.floor(normalized * num_bins) / num_bins
    quantized = quantized * (max_val - min_val) + min_val
    return quantized

@app.post("/face_scan", response_class=JSONResponse)
async def face_scan(face_image: UploadFile = File(...)):
    """Extract face embedding from provided image"""
    try:
        # Read image bytes
        contents = await face_image.read()

        # Preprocess image
        img = preprocess_image(contents)

        # Detect faces
        faces = face_analyzer.get(img)

        if not faces:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}). Using the most prominent one.")

        # Select the largest face
        face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]))

        # Extract augmented embeddings
        all_embeddings = augment_and_extract_embeddings(img, face)

        # Normalize and quantize embedding
        final_embedding = normalize_embedding(all_embeddings)

        # Convert to list for JSON response
        return final_embedding.tolist()

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing face scan: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face scan failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)