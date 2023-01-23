import os
import torch
import matplotlib.pyplot as plt
from flask import *
from wgan import NetG

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        n = int(request.form['n'])
        model = str(request.form['model'])
        if not os.path.exists("static"):
            os.mkdir("static")
        image_path = f"static/{model}_image.jpg"
        if model in "WGAN":
            generator_model = torch.load("Pretrained/wgan_modelG.pth")
            batch_size = n ** 2
            inputs = torch.randn(batch_size, 128)
            outputs = generator_model(inputs)

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
        elif model in "WGAN-GP":
            generator_model = torch.load("Pretrained/wgangp_modelG.pth")
            batch_size = n ** 2
            inputs = torch.randn(batch_size, 128)
            outputs = generator_model(inputs)

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

        return render_template("index.html", view_image=True, model=model, n=n)
    return render_template("index.html", view_image=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
