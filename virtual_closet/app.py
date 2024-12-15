import os
import sqlite3
import random
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Set upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database initialization
def init_db():
    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS clothes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_type TEXT NOT NULL,
            image_path TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
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

# Update clothes table to add user_id column if missing
def update_clothes_table():
    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(clothes)")
    columns = [column[1] for column in cursor.fetchall()]
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE clothes ADD COLUMN user_id INTEGER NOT NULL DEFAULT 1")
        print("user_id column added to clothes table.")

    conn.commit()
    conn.close()

init_db()
update_clothes_table()

# User registration
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

# User login
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

# User logout
@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

# Home page (only for logged-in users)
@app.route("/", methods=["GET", "POST"])
def index():
    if "user_id" not in session:
        flash("Please log in to access your virtual closet.", "warning")
        return redirect(url_for("login"))

    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()

    if request.method == "POST":
        files = request.files.getlist('file')
        item_type = request.form.get('category')

        if not item_type:
            return "No item type selected"

        for file in files:
            if file.filename == '':
                return "No selected file"

            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            cursor.execute("INSERT INTO clothes (item_type, image_path, user_id) VALUES (?, ?, ?)",
                           (item_type, filepath, session["user_id"]))

        conn.commit()
        return redirect(url_for("index"))

    # Fetch user's items
    cursor.execute("SELECT * FROM clothes WHERE user_id = ?", (session["user_id"],))
    clothes = cursor.fetchall()

    tops = [(item[0], item[2]) for item in clothes if item[1] == 'Top']
    bottoms = [(item[0], item[2]) for item in clothes if item[1] == 'Bottom']
    shoes = [(item[0], item[2]) for item in clothes if item[1] == 'Shoes']

    suggested_outfit = []
    if tops:
        suggested_outfit.append(random.choice(tops)[1])
    if bottoms:
        suggested_outfit.append(random.choice(bottoms)[1])
    if shoes:
        suggested_outfit.append(random.choice(shoes)[1])

    conn.close()

    return render_template("index.html", tops=tops, bottoms=bottoms, shoes=shoes, suggested_outfit=suggested_outfit)

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

# Add to wishlist
@app.route("/add_to_wishlist/<int:item_id>", methods=["POST"])
def add_to_wishlist(item_id):
    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM wishlist WHERE item_id = ? AND user_id = ?", (item_id, session["user_id"]))
    exists = cursor.fetchone()
    if not exists:
        cursor.execute("INSERT INTO wishlist (item_id, user_id) VALUES (?, ?)", (item_id, session["user_id"]))
        conn.commit()

    conn.close()
    return redirect(url_for("index"))

# View wishlist
@app.route("/wishlist")
def wishlist():
    if "user_id" not in session:
        flash("Please log in to access your wishlist.", "warning")
        return redirect(url_for("login"))

    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()

    cursor.execute('''
        SELECT c.id, c.item_type, c.image_path 
        FROM wishlist w
        JOIN clothes c ON w.item_id = c.id
        WHERE w.user_id = ?
    ''', (session["user_id"],))
    wishlist_items = cursor.fetchall()
    conn.close()

    return render_template("wishlist.html", wishlist_items=wishlist_items)

# Remove from wishlist
@app.route("/remove_from_wishlist/<int:item_id>", methods=["POST"])
def remove_from_wishlist(item_id):
    conn = sqlite3.connect('clothes.db')
    cursor = conn.cursor()
    cursor.execute("DELETE FROM wishlist WHERE item_id = ? AND user_id = ?", (item_id, session["user_id"]))
    conn.commit()
    conn.close()
    return redirect(url_for("wishlist"))

if __name__ == "__main__":
    app.run(debug=True)

