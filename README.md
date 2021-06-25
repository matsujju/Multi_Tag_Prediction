# [Multi_Tag_Recommendation](https://multitag.herokuapp.com/)

This repository contains files for this end-to-end Project (From Scraping to WebApp).
The app is made with [Dash](https://plotly.com/dash/) interactive python framework developed by [Plotly](https://plotly.com/).
Dash is a simple and effective to bind user interface around python code.
## Overview -
  * The Project is an end-to-end [WebApp](https://multitag.herokuapp.com/) built for interactive visualization of Machine learning Model used for classifying multiple tags from a given text input.
  * This Project also include [Preprocessing](https://github.com/matsujju/Multi_Tag_Prediction/blob/main/modelling.ipynb) and [Model Building](https://github.com/matsujju/Multi_Tag_Prediction/blob/main/stack-overflow-basic-preprocessing.ipynb) steps in the form of Jupyter Notebook where all the steps from Basic cleaning of text to Building a accurate model are present.
  * It also includes text data from StackOverflow Website which used Scrapy Spiders to scrape and Collect the data of 200,000+ Questions and Answers and can be found here.
  
  
## Getting Started -
### Running the app locally
First create a virtual environment with conda or venv inside a temp folder, then activate it.
```
virtualenv venv

# Windows
venv\Scripts\activate
# Or Linux
source venv/bin/activate

```
Clone the git repo, then install the requirements with pip
```
git clone https://github.com/matsujju/Multi_Tag_Prediction.git
cd Desktop/temp_folder/Multi_Tag_Prediction/        (Here temp_folder is in Desktop...choose your own path if different)
pip install -r requirements.txt
```
Run the app (from your terminal)
```
python dash_app.py
```
Open a browser at http://127.0.0.1:8050

## About the app -
This WebApp predicts the tags based on the user input, given the input is Computer Programming related. It lets user select some preprocessing functions, number of tags and threshold value to play with.

## Built with -
  * [Dash](https://dash.plotly.com/) - Main server and interactive components
  * [Plotly Python](https://plotly.com/python/) - Used to create the interactive plots
  * [Pandas](https://pandas.pydata.org/) - Exploring and Manipulating the data
  * [scikit-learn](https://scikit-learn.org/stable/) - Simple library for predictive data analysis
  * [NLTK](https://www.nltk.org/) - Library to work with human language data
  
## Screenshots -
Followings are the screenshots of the app in this repo:

![](https://github.com/matsujju/Multi_Tag_Prediction/blob/main/screenshots/webm.gif)
![](https://github.com/matsujju/Multi_Tag_Prediction/blob/main/screenshots/Untitled1.png)
![](https://github.com/matsujju/Multi_Tag_Prediction/blob/main/screenshots/Untitled.png)

## Credits -
  * Looking for more visualized dash apps? See [here](https://dash-gallery.plotly.host/Portal/)
  * [Dash Documentation](https://dash.plotly.com/introduction)
  * [Plotly Python Documentation](https://plotly.com/python/)
  * [Video tutorials](https://www.youtube.com/channel/UCqBFsuAz41sqWcFjZkqmJqQ)
  * [Awesome Community](https://community.plotly.com/) for helping
