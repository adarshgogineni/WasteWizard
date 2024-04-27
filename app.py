import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired
from flask import jsonify
from flask import flash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg'}

login_manager = LoginManager()
login_manager.init_app(app)

# temporary database
users = {
    'user1': {'password': 'pass1', 'first_name': 'John', 'last_name': 'Doe', 'items_recycled': 5},
    'user2': {'password': 'pass2', 'first_name': 'Jane', 'last_name': 'Doe', 'items_recycled': 3}
}


#check file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class User(UserMixin):
    def __init__(self, username, first_name, last_name, items_recycled):
        self.id = username
        self.first_name = first_name
        self.last_name = last_name
        self.items_recycled = items_recycled

@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        user_info = users[user_id]
        return User(user_id, user_info['first_name'], user_info['last_name'], user_info['items_recycled'])
    return None

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    first_name = StringField('First Name', validators=[InputRequired()])
    last_name = StringField('Last Name', validators=[InputRequired()])

@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    error = None
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        if username in users and users[username]['password'] == password:
            user_info = users[username]
            user = User(username, user_info['first_name'], user_info['last_name'], user_info['items_recycled'])
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password'
    return render_template('login.html', form=form, error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        users[form.username.data] = {
            'password': form.password.data,
            'first_name': form.first_name.data,
            'last_name': form.last_name.data,
            'items_recycled': 0
        }
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload_image', methods=['POST'])
@login_required
def upload_image():
    file = request.files.get('file')  # This is a safer way to access the files
    if file:
        filepath = os.path.join('uploads', secure_filename(file.filename))
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully!'})
    return jsonify({'message': 'No file found'}), 400


@app.route('/Profile')
@login_required
def profile():
    user_info = users[current_user.id]
    print(user_info)
    return render_template('Profile.html', user=user_info, username=current_user.id)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
