import os
import torch
import matplotlib.pyplot as plt
from flask import *
from wgan import *

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        n = int(request.form['n'])
        gender = str(request.form['gender'])
        if not os.path.exists("static"):
            os.mkdir("static")
        image_path = f"static/wgangp_image.jpg"
        generator_model = torch.load("Pretrained/wgangp_modelG.pth")

        if gender in "Female":
            batch_size = n ** 2
            inputs = torch.randn(batch_size, 128)
            labels = torch.randint(low=0, high=1, size=(batch_size,))
            outputs = generator_model(inputs, labels)

            plt.figure()
            for i in range(1, batch_size + 1):
                image = 0.5 + outputs[i - 1] * 0.5
                image = image * 255.0
                image = image.permute(1, 2, 0)
                image = image.byte().numpy()
                plt.subplot(n, n, i)
                plt.axis('off')
                plt.imshow(image)

            plt.tight_layout()
            plt.savefig(image_path)

        elif gender in "Male":
            batch_size = n ** 2
            inputs = torch.randn(batch_size, 128)
            labels = torch.randint(low=1, high=2, size=(batch_size,))
            outputs = generator_model(inputs, labels)

            plt.figure()
            for i in range(1, batch_size + 1):
                image = 0.5 + outputs[i - 1] * 0.5
                image = image * 255.0
                image = image.permute(1, 2, 0)
                image = image.byte().numpy()
                plt.subplot(n, n, i)
                plt.axis('off')
                plt.imshow(image)

            plt.tight_layout()
            plt.savefig(image_path)

        elif gender in "Random":
            batch_size = n ** 2
            inputs = torch.randn(batch_size, 128)
            labels = torch.randint(low=0, high=2, size=(batch_size,))
            outputs = generator_model(inputs, labels)

            plt.figure()
            for i in range(1, batch_size + 1):
                image = 0.5 + outputs[i - 1] * 0.5
                image = image * 255.0
                image = image.permute(1, 2, 0)
                image = image.byte().numpy()
                plt.subplot(n, n, i)
                plt.axis('off')
                plt.imshow(image)

            plt.tight_layout()
            plt.savefig(image_path)

        return render_template("index.html", view_image=True, gender=gender, n=n)
    return render_template("index.html", view_image=False)

@app.route("/image/<n>/<gender>")
def image(n, gender):
    n = int(n)
    gender = int(gender)
    if not os.path.exists("static"):
        os.mkdir("static")
    image_path = f"static/wgangp_image.jpg"
    generator_model = torch.load("Pretrained/wgangp_modelG.pth")

    if gender == 1:
        batch_size = n ** 2
        inputs = torch.randn(batch_size, 128)
        labels = torch.randint(low=0, high=1, size=(batch_size,))
        outputs = generator_model(inputs, labels)

        plt.figure()
        for i in range(1, batch_size + 1):
            image = 0.5 + outputs[i - 1] * 0.5
            image = image * 255.0
            image = image.permute(1, 2, 0)
            image = image.byte().numpy()
            plt.subplot(n, n, i)
            plt.axis('off')
            plt.imshow(image)

        plt.tight_layout()
        plt.savefig(image_path)

    elif gender == 2:
        batch_size = n ** 2
        inputs = torch.randn(batch_size, 128)
        labels = torch.randint(low=1, high=2, size=(batch_size,))
        outputs = generator_model(inputs, labels)

        plt.figure()
        for i in range(1, batch_size + 1):
            image = 0.5 + outputs[i - 1] * 0.5
            image = image * 255.0
            image = image.permute(1, 2, 0)
            image = image.byte().numpy()
            plt.subplot(n, n, i)
            plt.axis('off')
            plt.imshow(image)

        plt.tight_layout()
        plt.savefig(image_path)

    elif gender == 3:
        batch_size = n ** 2
        inputs = torch.randn(batch_size, 128)
        labels = torch.randint(low=0, high=2, size=(batch_size,))
        outputs = generator_model(inputs, labels)

        plt.figure()
        for i in range(1, batch_size + 1):
            image = 0.5 + outputs[i - 1] * 0.5
            image = image * 255.0
            image = image.permute(1, 2, 0)
            image = image.byte().numpy()
            plt.subplot(n, n, i)
            plt.axis('off')
            plt.imshow(image)

        plt.tight_layout()
        plt.savefig(image_path)
    return render_template("image.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
