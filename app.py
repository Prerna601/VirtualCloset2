import os
import sqlite3
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from train_model import FashionCNN  # Import the trained AI model class

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Set upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🔹 **STEP 1: Load the AI Model for Image Classification**
# Load class labels (same as during training)
class_labels = ['Blazer', 'Celana_Panjang', 'Celana_Pendek', 'Gaun', 'Hoodie', 
                'Jaket', 'Jaket_Denim', 'Jaket_Olahraga', 'Jeans', 'Kaos', 
                'Kemeja', 'Mantel', 'Polo', 'Rok','Shoes', 'Sweter']

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize image to match model input
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Same normalization as training
])

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionCNN(num_classes=len(class_labels)).to(device)
model.load_state_dict(torch.load("fashion_classifier.pth", map_location=device))
model.eval()  # Set model to evaluation mode

# Function to predict image category
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        predicted_class = torch.argmax(output).item()

    return class_labels[predicted_class]  # e.g., "Jeans"

# def predict_image(image_path):
#     img = Image.open(image_path).convert("RGB")  # Ensure image is RGB
#     img = transform(img).unsqueeze(0)  # Add batch dimension
#     img = img.to(device)

#     with torch.no_grad():
#         output = model(img)
#         predicted_class = torch.argmax(output).item()
#         label = class_labels[predicted_class]

#     # 🔽 Map AI class to UI category
#     label_to_category = {
#         "Kaos": "Top", "Kemeja": "Top", "Hoodie": "Top", "Jaket": "Top", "Blazer": "Top",
#         "Sweter": "Top", "Polo": "Top", "Mantel": "Top", "Jaket_Denim": "Top", "Jaket_Olahraga": "Top",
#         "Rok": "Bottom", "Celana_Panjang": "Bottom", "Celana_Pendek": "Bottom", "Jeans": "Bottom",
#         "Gaun": "Bottom", "Shoes": "Shoes"
#     }

#     return label_to_category.get(label, "Top")  # fallback to Top if unknown


# Database initialization
def init_db():
    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()

    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')

    # Clothes table (Modified to include attributes column)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clothes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,
            image_path TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            attributes TEXT,  -- New column for AI-based tagging
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Wishlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wishlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            FOREIGN KEY (item_id) REFERENCES clothes (id),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()

init_db()

# 🔹 **STEP 2: Modify the Image Upload Route for AI Categorization**
# @app.route("/upload", methods=["POST"])
# def upload():
#     if "user_id" not in session:
#         flash("Please log in first.", "warning")
#         return redirect(url_for("login"))

#     if "file" not in request.files:
#         flash("No file part", "danger")
#         return redirect(url_for("index"))

#     file = request.files["file"]

#     if file.filename == "":
#         flash("No selected file", "danger")
#         return redirect(url_for("index"))

#     filename = secure_filename(file.filename)
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     file.save(file_path)
#     print(f"📂 Attempting to save file at: {file_path}")
#     print(f"✅ File successfully saved at: {file_path}")

#     # 🔽 Predict label using AI
#     predicted_label = predict_image(file_path)

#     # 🔽 Map AI label (e.g., "Kaos") to general category
#     label_to_category = {
#         "Kaos": "Top", "Kemeja": "Top", "Hoodie": "Top", "Jaket": "Top", "Blazer": "Top",
#         "Sweter": "Top", "Polo": "Top", "Mantel": "Top", "Jaket_Denim": "Top", "Jaket_Olahraga": "Top",
#         "Rok": "Bottom", "Celana_Panjang": "Bottom", "Celana_Pendek": "Bottom", "Jeans": "Bottom",
#         "Gaun": "Bottom", "Shoes": "Shoes"
#     }

#     category = label_to_category.get(predicted_label, "Top")  # fallback to Top

#     # 🔽 Save using mapped category
#     conn = sqlite3.connect('clothes.db')
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO clothes (item_type, image_path, user_id) VALUES (?, ?, ?)",
#                    (category, file_path, session["user_id"]))
#     conn.commit()
#     print("🧠 Stored to DB:", category, file_path, session["user_id"])
#     conn.close()

#     flash(f"✅ Uploaded and categorized as: {predicted_label} → {category}", "success")
#     return redirect(url_for("index"))
@app.route("/upload", methods=["POST"])
def upload():
    if "user_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    if "file" not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename).replace("\\", "/")
    file.save(file_path)
    print(f"📂 Saved file at: {file_path}")

    predicted_label = predict_image(file_path)  # e.g., "Jeans"

    # 🔁 Map AI label to general category
    label_to_category = {
        "Kaos": "Top", "Kemeja": "Top", "Hoodie": "Top", "Jaket": "Top", "Blazer": "Top",
        "Sweter": "Top", "Polo": "Top", "Mantel": "Top", "Jaket_Denim": "Top", "Jaket_Olahraga": "Top",
        "Rok": "Bottom", "Celana_Panjang": "Bottom", "Celana_Pendek": "Bottom", "Jeans": "Bottom",
        "Gaun": "Bottom", "Shoes": "Shoes"
    }

    category = label_to_category.get(predicted_label, "Top")  # fallback to Top
    print(f"🧠 Predicted: {predicted_label} → Category: {category}")

    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO clothes (item_type, image_path, user_id) VALUES (?, ?, ?)",
                   (category, file_path, session["user_id"]))
    conn.commit()
    conn.close()

    flash(f"✅ Uploaded and categorized as: {predicted_label} → {category}", "success")
    return redirect(url_for("index"))




# 🔹 Fix `/add_to_wishlist` Error
@app.route("/add_to_wishlist/<int:item_id>", methods=["POST"])
def add_to_wishlist(item_id):
    if "user_id" not in session:
        flash("Please log in first.", "warning")
        return redirect(url_for("login"))

    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()

    # Check if item exists
    cursor.execute("SELECT id FROM clothes WHERE id = ?", (item_id,))
    item = cursor.fetchone()
    if not item:
        flash("Item not found!", "danger")
        conn.close()
        return redirect(url_for("index"))

    # Check if item is already in wishlist
    cursor.execute("SELECT * FROM wishlist WHERE item_id = ? AND user_id = ?", (item_id, session["user_id"]))
    exists = cursor.fetchone()

    if not exists:
        cursor.execute("INSERT INTO wishlist (item_id, user_id) VALUES (?, ?)", (item_id, session["user_id"]))
        conn.commit()
        flash("Item added to wishlist!", "success")
    else:
        flash("Item already in wishlist!", "info")

    conn.close()
    return redirect(url_for("index"))

# 🔹 Fix Missing Image Display
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")

        conn = sqlite3.connect('clothes.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists. Please try again.", "danger")
        finally:
            conn.close()

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect('clothes.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[2], password):
            session["user_id"] = user[0]
            session["username"] = user[1]
            flash("Login successful!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password.", "danger")

    return render_template("login.html")


# Home page (only for logged-in users)
@app.route("/", methods=["GET", "POST"])
def index():
    if "user_id" not in session:
        flash("Please log in to access your virtual closet.", "warning")
        return redirect(url_for("login"))

    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()

    # Fetch all clothes for the user
    cursor.execute("SELECT * FROM clothes WHERE user_id = ?", (session["user_id"],))
    clothes = cursor.fetchall()
    conn.close()

    # Split into Top, Bottom, Shoes
    tops = [(item[0], item[2]) for item in clothes if item[1] == 'Top']
    bottoms = [(item[0], item[2]) for item in clothes if item[1] == 'Bottom']
    shoes = [(item[0], item[2]) for item in clothes if item[1] == 'Shoes']

    return render_template("index.html", tops=tops, bottoms=bottoms, shoes=shoes)


# Delete item
@app.route("/delete/<int:item_id>", methods=["POST"])
def delete_item(item_id):
    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()
    cursor.execute("SELECT image_path FROM clothes WHERE id = ? AND user_id = ?", (item_id, session["user_id"]))
    image = cursor.fetchone()

    if image:
        image_path = image[0]
        if os.path.exists(image_path):
            os.remove(image_path)

        cursor.execute("DELETE FROM clothes WHERE id = ?", (item_id,))
        conn.commit()

    conn.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
