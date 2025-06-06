# Number Guess

Foundation project for admission into the [MLI AI residency](https://programme.mlx.institute/about).

Consists of a PyTorch Convolutional Neural Network trained on the MNIST dataset to recognise a single hand-drawn digit, plus a Streamlit web app to doodle a digit on a cavas, and log the prediction result into a PostgreSQL database.

I started by going through this PyTorch tutorial to do just that:
https://www.datatechnotes.com/2024/04/mnist-image-classification-with-pytorch.html

After sucessfully running the tutorial code and generating the model file (`.pth`) on my machine, I started breaking down the code and extracting what was needed to run a webapp without worrying about model training and evaluation.

Then I started looking into Streamlit and was pleasantly suprised by it's simplicty. I do JavaScript and React for a living and Python still feels foreign, but it's very fun to use after you get past the initial ecosystem setup.

Next up, it would be cool to feed the user doodles back into the model for fine-tuning.

## Usage

After installing docker with docker-compose, create your own .env file with

```
cp .env.example .env
```

and then run

```
docker compose up
```

## Live demo

If the service hasn't crashed for some reason, you can check it at http://tao.cl:8501/
