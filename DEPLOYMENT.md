# Streamlit Cloud Deployment Instructions

This document provides instructions for deploying the Brain Tumor Classification app to Streamlit Cloud.

## Requirements

- A GitHub repository containing your code
- A Streamlit Cloud account

## Deployment Steps

1. **Fork or push your code to GitHub**
   
   Make sure your code is available in a GitHub repository.

2. **Use the correct requirements file**

   For Streamlit Cloud deployment, use `requirements-cloud.txt` which excludes 
   Apple Silicon-specific packages like `tensorflow-metal`.

3. **Deploy on Streamlit Cloud**

   - Log in to [Streamlit Cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Connect your GitHub repository
   - Set the main file path to `app.py`
   - Advanced Settings:
     - Set Python version to 3.11
     - Set requirements file to `requirements-cloud.txt`
   - Click "Deploy"

## Troubleshooting

- If you get errors about packages not being found, check the deployment logs
- Make sure you're using `requirements-cloud.txt` and not the regular `requirements.txt`
- The app will automatically detect it's running on Streamlit Cloud and use CPU mode

## Local Development vs Cloud Deployment

- Local development on macOS will use Metal GPU acceleration when available
- Cloud deployment will always use CPU mode for compatibility
- The `deployment_config.py` file handles these differences automatically
