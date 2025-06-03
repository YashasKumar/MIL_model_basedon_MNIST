# MIL on MNIST – Bag-Level Percentage Prediction API

This project implements a Multiple Instance Learning (MIL) pipeline from scratch to predict the **percentage of digit '0's** in bags of MNIST images. Using a weakly supervised learning approach, the model processes bags of digit images filtered to only digits 0 and 7, and estimates the proportion of zeros in each bag.

---

## Project Overview

Multiple Instance Learning (MIL) is a powerful technique in weakly supervised learning where labels are available only at the bag level instead of individual instances. Here, each bag contains multiple MNIST digit images, and the goal is to predict the fraction of zeros present in that bag.

This project includes:
- Custom MIL model architecture implemented from scratch.
- Data preparation creating bags from MNIST digits filtered to 0s and 7s.
- A FastAPI backend exposing endpoints to evaluate the model and visualize predictions.

---

## Usage Example

- Predict the percentage of zeros in given MNIST bags.
- Retrieve model evaluation metrics.
- Visualize prediction performance via saved plots.

---

## Live Demo

Try the API live at [this link](https://mnist-mil-service-573238441616.us-central1.run.app/)

---

## Deployment Notes

The API is containerized with Docker and deployed on Google Cloud Run with appropriate resource settings to handle the model’s memory requirements. The service automatically scales based on demand.

---

## Author

**Yashas Kumar S**  
Independent project exploring weak supervision with Multiple Instance Learning on MNIST.
