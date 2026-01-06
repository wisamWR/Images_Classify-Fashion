import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps


# --- Page Config ---
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="🧥",
    layout="wide"
)

# --- Styles ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('fashion_model.h5')
        return model
    except Exception as e:
        return None

def preprocess_image(image):
    # 1. Convert to grayscale
    image = ImageOps.grayscale(image)
    
    # 2. Resize to 28x28
    image = image.resize((28, 28))
    
    # 3. Invert colors? 
    # Fashion MNIST has black background (0) and white content (255).
    # Real photos usually have white background and dark clothes.
    # We check the average pixel value. If high (bright/white), we invert.
    img_array = np.array(image)
    if img_array.mean() > 127: # Assumes white background
        image = ImageOps.invert(image)
    
    # 4. Convert to array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    # 5. Reshape for model (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array, image

# --- Main App ---
def main():
    st.title("🧥 Fashion MNIST AI Classifier")
    st.markdown("### Unggah gambar pakaian Anda untuk diklasifikasikan!")

    # Sidebar
    with st.sidebar:
        st.header("Tentang Aplikasi")
        st.info(
            "Aplikasi ini menggunakan **Convolutional Neural Network (CNN)** "
            "yang dilatih pada dataset Fashion MNIST."
        )
        st.markdown("**Kelas yang dikenali:**")
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        st.markdown("\n".join([f"- {c}" for c in class_names]))

    # Model Loading
    model = load_model()
    if model is None:
        st.error("Model 'fashion_model.h5' tidak ditemukan!")
        st.warning("Silakan jalankan script `train_model.py` terlebih dahulu untuk melatih dan menyimpan model.")
        if st.button("Latih Model Sekarang (Mungkin butuh waktu!)"):
            with st.spinner("Sedang melatih model..."):
                import train_model
                train_model.train_and_save_model()
                st.success("Model berhasil dilatih! Silakan refresh halaman.")
        return

    # Layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload Gambar")
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar yang diunggah', use_column_width=True)

    with col2:
        st.subheader("2. Hasil Klasifikasi")
        if uploaded_file is not None:
            if st.button("🔍 Analisis Gambar"):
                with st.spinner('Sedang menganalisis...'):
                    # Preprocess
                    processed_data, processed_image = preprocess_image(image)
                    
                    # Debug: Show processed image (what the model sees)
                    st.write("Input Model (28x28 Grayscale):")
                    st.image(processed_image, width=100)

                    # Predict
                    predictions = model.predict(processed_data)
                    score = tf.nn.softmax(predictions[0]) # Optional if model output is already softmax
                    
                    # Get results
                    predicted_class_idx = np.argmax(predictions[0])
                    predicted_class = class_names[predicted_class_idx]
                    confidence = np.max(predictions[0]) * 100

                    # Display
                    st.success(f"Prediksi: **{predicted_class}**")
                    st.metric("Tingkat Keyakinan", f"{confidence:.2f}%")

                    # Chart
                    st.write("Distribusi Probabilitas:")
                    chart_data = {class_names[i]: float(predictions[0][i]) for i in range(10)}
                    st.bar_chart(chart_data)

if __name__ == "__main__":
    main()
