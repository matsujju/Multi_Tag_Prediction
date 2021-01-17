# Let's start
import dash
from dash_bootstrap_components._components.FormGroup import FormGroup
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from bs4 import BeautifulSoup
import unicodedata
import spacy
import base64

nlp = spacy.load("en_core_web_sm")

from nltk.tokenize import word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer

nltk.download("stopwords")
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words("english")

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

################################################################################
def data_clean(text):
    text = text.lower()  # lowering of characters
    text = text.strip()  # removeing extra spaces before and after the sentence
    # text = TextBlob(text).correct() # spelling mistakes correction  (very slow) (maybe try directly on dataframe column)
    return text


# remove accented characters (Sómě Áccěntěd těxt)
def remove_accented_chars(text):
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return text


def strip_html_tags(text):  # function for removing html tags
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


def remove_special_characters(
    text, remove_digits=False
):  # function for removing punctuations , before using this remove any contractions(like "I'll" --> "I will") in text data
    pattern = r"[^a-zA-z0-9\s\-\+\#]" if not remove_digits else r"[^a-zA-z\s\-\+\#]"
    text = re.sub(pattern, "", text)
    return text


def remove_links_emojis(text):
    pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )  # removing https:www.examples.com
    text = pattern.sub("", text)

    emoji = re.compile(
        "["
        "\U0001F600-\U0001FFFF"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji.sub(r"", text)
    return text


def contractions(text):  # contraction of text i.e remove_apostrophes
    text = re.sub("ain't", "is not", str(text))
    text = re.sub("aren't", "are not", str(text))
    text = re.sub("can't", "cannot", str(text))
    text = re.sub("can't've", "cannot have", str(text))
    text = re.sub("'cause", "because", str(text))
    text = re.sub("could've", "could have", str(text))
    text = re.sub("couldn't", "could not", str(text))
    text = re.sub("couldn't've", "could not have", str(text))
    text = re.sub("didn't", "did not", str(text))
    text = re.sub("doesn't", "does not", str(text))
    text = re.sub("don't", "do not", str(text))
    text = re.sub("hadn't", "had not", str(text))
    text = re.sub("hadn't've", "had not have", str(text))
    text = re.sub("hasn't", "has not", str(text))
    text = re.sub("haven't", "have not", str(text))
    text = re.sub("he'd", "he would", str(text))
    text = re.sub("he'd've", "he would have", str(text))
    text = re.sub("he'll", "he will", str(text))
    text = re.sub("he'll've", "he he will have", str(text))
    text = re.sub("he's", "he is", str(text))
    text = re.sub("how'd", "how did", str(text))
    text = re.sub("how'd'y", "how do you", str(text))
    text = re.sub("how'll", "how will", str(text))
    text = re.sub("how's", "how is", str(text))
    text = re.sub("I'd", "I would", str(text))
    text = re.sub("I'd've", "I would have", str(text))
    text = re.sub("I'll", "I will", str(text))
    text = re.sub("I'll've", "I will have", str(text))
    text = re.sub("I'm", "I am", str(text))
    text = re.sub("I've", "I have", str(text))
    text = re.sub("i'd", "i would", str(text))
    text = re.sub("i'd've", "i would have", str(text))
    text = re.sub("i'll", "i will", str(text))
    text = re.sub("i'll've", "i will have", str(text))
    text = re.sub("i'm", "i am", str(text))
    text = re.sub("i've", "i have", str(text))
    text = re.sub("isn't", "is not", str(text))
    text = re.sub("it'd", "it would", str(text))
    text = re.sub("it'd've", "it would have", str(text))
    text = re.sub("it'll", "it will", str(text))
    text = re.sub("it'll've", "it will have", str(text))
    text = re.sub("it's", "it is", str(text))
    text = re.sub("let's", "let us", str(text))
    text = re.sub("ma'am", "madam", str(text))
    text = re.sub("mayn't", "may not", str(text))
    text = re.sub("might've", "might have", str(text))
    text = re.sub("mightn't", "might not", str(text))
    text = re.sub("mightn't've", "might not have", str(text))
    text = re.sub("must've", "must have", str(text))
    text = re.sub("mustn't", "must not", str(text))
    text = re.sub("mustn't've", "must not have", str(text))
    text = re.sub("needn't", "need not", str(text))
    text = re.sub("needn't've", "need not have", str(text))
    text = re.sub("o'clock", "of the clock", str(text))
    text = re.sub("oughtn't", "ought not", str(text))
    text = re.sub("oughtn't've", "ought not have", str(text))
    text = re.sub("shan't", "shall not", str(text))
    text = re.sub("sha'n't", "shall not", str(text))
    text = re.sub("shan't've", "shall not have", str(text))
    text = re.sub("she'd", "she would", str(text))
    text = re.sub("she'd've", "she would have", str(text))
    text = re.sub("she'll", "she will", str(text))
    text = re.sub("she'll've", "she will have", str(text))
    text = re.sub("she's", "she is", str(text))
    text = re.sub("should've", "should have", str(text))
    text = re.sub("shouldn't", "should not", str(text))
    text = re.sub("shouldn't've", "should not have", str(text))
    text = re.sub("so've", "so have", str(text))
    text = re.sub("so's", "so as", str(text))
    text = re.sub("that'd", "that would", str(text))
    text = re.sub("that'd've", "that would have", str(text))
    text = re.sub("that's", "that is", str(text))
    text = re.sub("there'd", "there would", str(text))
    text = re.sub("there'd've", "there would have", str(text))
    text = re.sub("there's", "there is", str(text))
    text = re.sub("they'd", "they would", str(text))
    text = re.sub("they'd've", "they would have", str(text))
    text = re.sub("they'll", "they will", str(text))
    text = re.sub("they'll've", "they will have", str(text))
    text = re.sub("they're", "they are", str(text))
    text = re.sub("they've", "they have", str(text))
    text = re.sub("to've", "to have", str(text))
    text = re.sub("wasn't", "was not", str(text))
    text = re.sub("we'd", "we would", str(text))
    text = re.sub("we'd've", "we would have", str(text))
    text = re.sub("we'll", "we will", str(text))
    text = re.sub("we'll've", "we will have", str(text))
    text = re.sub("we're", "we are", str(text))
    text = re.sub("we've", "we have", str(text))
    text = re.sub("weren't", "were not", str(text))
    text = re.sub("what'll", "what will", str(text))
    text = re.sub("what'll've", "what will have", str(text))
    text = re.sub("what're", "what are", str(text))
    text = re.sub("what's", "what is", str(text))
    text = re.sub("what've", "what have", str(text))
    text = re.sub("when's", "when is", str(text))
    text = re.sub("when've", "when have", str(text))
    text = re.sub("where'd", "where did", str(text))
    text = re.sub("where's", "where is", str(text))
    text = re.sub("where've", "where have", str(text))
    text = re.sub("who'll", "who will", str(text))
    text = re.sub("who'll've", "who will have", str(text))
    text = re.sub("who's", "who is", str(text))
    text = re.sub("who've", "who have", str(text))
    text = re.sub("why's", "why is", str(text))
    text = re.sub("why've", "why have", str(text))
    text = re.sub("will've", "will have", str(text))
    text = re.sub("won't", "will not", str(text))
    text = re.sub("won't've", "will not have", str(text))
    text = re.sub("would've", "would have", str(text))
    text = re.sub("wouldn't", "would not", str(text))
    text = re.sub("wouldn't've", "would not have", str(text))
    text = re.sub("y'all", "you all", str(text))
    text = re.sub("y'all'd", "you all would", str(text))
    text = re.sub("y'all'd've", "you all would have", str(text))
    text = re.sub("y'all're", "you all are", str(text))
    text = re.sub("y'all've", "you all have", str(text))
    text = re.sub("you'd", "you would", str(text))
    text = re.sub("you'd've", "you would have", str(text))
    text = re.sub("you'll", "you will", str(text))
    text = re.sub("you'll've", "you will have", str(text))
    text = re.sub("you're", "you are", str(text))
    text = re.sub("you've", "you have", str(text))
    return text


def lemmatize_text(text):  # function for lemmetization of text
    text = nlp(text)
    text = str(
        " ".join(
            [word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text]
        )
    )
    return text


def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = " ".join(filtered_tokens)
    return filtered_text


def preprocess(
    text,
    clean_data=True,
    contraction_expansion=True,
    accented_char_removal=False,
    lemmatize_the_text=False,
    strip_html_tag=True,
    special_characters=True,
    remove_digits=False,
    stop_words=False,
    remove_emoji_links=True,
):

    # cleaned data (lower & strip whitespaces & spelling mistake)
    if clean_data:
        text = data_clean(text)
    # strip html
    if strip_html_tag:
        text = strip_html_tags(text)
    # accented char removal
    if accented_char_removal:
        text = remove_accented_chars(text)
    if remove_emoji_links:
        text = remove_links_emojis(text)
    # exapand contraction
    if contraction_expansion:
        text = contractions(text)
    if stop_words:
        text = remove_stopwords(text)
    # lemmetization
    if lemmatize_the_text:
        text = lemmatize_text(text)
    # punctuations and digits
    if special_characters:
        text = remove_special_characters(text, remove_digits=remove_digits)

    # remove extra whitespace
    text = re.sub(" +", " ", text)

    return text


classifier = joblib.load("logreg100_multi_label2.pkl.gz")
multilabel = joblib.load("multilabel_binarizer_3.pkl.gz")

tfidf_vocab = joblib.load("tfidf_file_100k.pkl.gz")
tfidf = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 4),
    stop_words="english",
    vocabulary=tfidf_vocab.vocabulary_,
)


def get_tags(predicted_list, threshold, labels):
    mlb = [(i1, c1) for i1, c1 in enumerate(multilabel.classes_)]
    temp_list = sorted(
        [(i, c) for i, c in enumerate(list(predicted_list))],
        key=lambda x: x[1],
        reverse=True,
    )
    tag_list = [item1 for item1 in temp_list if item1[1] >= threshold]
    tags = [
        item[1] for item2 in tag_list[:labels] for item in mlb if item2[0] == item[0]
    ]
    return tags


app = dash.Dash(
    __name__, title="Multi Tag Prediction", external_stylesheets=[dbc.themes.MINTY]
)
server = app.server
# image_filename = r"assets\dash-logo.png"
# encoded_image = base64.b64encode(open(image_filename, "rb").read()).decode("ascii")

app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            style={
                                "height": "55px",
                                "width": "auto%",
                                "margin-right": "2%",
                                "padding-top": "10px",
                                "padding-bottom": "0px",
                                "display": "inline-block",
                                "float": "left",
                                "margin-left": "3.25%",
                                # "margin-top": "2%",
                            },
                        )
                    ],
                ),
                html.Div(
                    [
                        html.H2(
                            ["Multi Tag Prediction", html.Hr(),],
                            style={
                                "text-align": "center",
                                "margin-left": "35%",
                                "margin-right": "2%",
                                "margin-bottom": "5%",
                                "margin-top": "2%",
                                "color": "Tomato",
                            },
                        ),
                    ],
                    style={"width": "50%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.A(
                            dbc.Button(
                                "Learn More",
                                id="learn-more-button",
                                outline=True,
                                color="info",
                            ),
                            href="https://github.com/matsujju/Multi_Tag_Prediction",
                        )
                    ],
                    style={
                        "width": "auto%",
                        "display": "inline-block",
                        "margin-left": "22.5%",
                    },
                    id="button",
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dbc.FormGroup(
                                [
                                    dbc.Label(
                                        "🌴 Filter by Preprocessing Functions to Apply:"
                                    ),
                                    dbc.RadioItems(
                                        options=[
                                            {"label": "All", "value": "all"},
                                            {"label": "Customized", "value": "custom"},
                                        ],
                                        value="all",
                                        id="radio-button",
                                        inline=True,
                                        style={"padding": "10px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    dbc.FormGroup(
                                        [
                                            dbc.Checklist(
                                                id="drop-down",
                                                options=[
                                                    {
                                                        "label": "remove_digits🔢",
                                                        "value": "remove_digits",
                                                    },
                                                    {
                                                        "label": "remove_accented_chars (Ñ)",
                                                        "value": "accented_char_removal",
                                                    },
                                                    {
                                                        "label": "lemmatize_the_text",
                                                        "value": "lemmatize_the_text",
                                                    },
                                                    {
                                                        "label": "remove_stopwords",
                                                        "value": "stop_words",
                                                    },
                                                ],
                                                value=[
                                                    "remove_digits",
                                                    "accented_char_removal",
                                                    "lemmatize_the_text",
                                                    "stop_words",
                                                ],
                                                switch=True,
                                            )
                                        ]
                                    )
                                ],
                                style={"padding": "5px", "margin-bottom": "10px",},
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "🌴 Filter by threshold (probabilty of labels to include):"
                                    ),
                                    dcc.Slider(
                                        id="slider",
                                        min=0.1,
                                        max=0.9,
                                        step=0.1,
                                        value=0.5,
                                        marks={
                                            0.1: "0.1",
                                            0.2: "0.2",
                                            0.3: "0.3",
                                            0.4: "0.4",
                                            0.5: "0.5",
                                            0.6: "0.6",
                                            0.7: "0.7",
                                            0.8: "0.8",
                                            0.9: "0.9",
                                        },
                                        tooltip={"placement": "top"},
                                        included=False,
                                    ),
                                ],
                                style={"margin-bottom": "30px"},
                            ),
                            html.Div(
                                [
                                    html.H6(
                                        "🌴 Filter by label count (how many labels to include if there are more):"
                                    ),
                                    dcc.Slider(
                                        id="slider-2",
                                        min=5,
                                        max=10,
                                        step=1,
                                        value=5,
                                        marks={
                                            5: "5",
                                            6: "6",
                                            7: "7",
                                            8: "8",
                                            9: "9",
                                            10: "10",
                                        },
                                        tooltip={"placement": "top"},
                                        included=False,
                                    ),
                                ],
                                style={"margin-top": "30px"},
                            ),
                        ],
                        style={
                            "width": "100%",
                            "box-shadow": "5px 5px 5px  #cacfd2",
                            "padding": "35px",
                            "background-color": "#f9f9f9",
                            "margin-left": "12.5%",
                            "margin-right": "7.5%",
                            "height": "99.5%",
                            "border-radius": "8px",
                            "border": "1px solid #ffbe42",
                        },
                    ),
                    width={"size": 4, "order": "first", "offset": 0},
                ),
                dbc.Col(
                    html.Div(
                        [
                            dbc.Textarea(
                                id="input_text",
                                value="",
                                style={
                                    "height": 300,
                                    "width": "140%",
                                    "box-shadow": "5px 5px 5px  #cacfd2",
                                    "background-color": "#f9f9f9",
                                    "border-radius": "8px",
                                    "border": "1px solid #ffbe42",
                                },
                                maxLength=10000,
                                placeholder="⌨️ Type the text or copy-paste from somewhere else",
                            ),
                            dbc.Button(
                                "Submit",
                                id="submit-button",
                                n_clicks=0,
                                style={"margin-left": "625px", "margin-top": "15px",},
                                outline=True,
                                color="primary",
                            ),
                            html.Div(
                                [
                                    dbc.Spinner(
                                        [
                                            # dbc.Toast(
                                            #     id="output-state",
                                            #     children=[],
                                            #     header="Predicted Labels 🏷️:",
                                            #     icon="info",
                                            #     is_open=True,
                                            #     duration=60000,
                                            #     dismissable=False,
                                            #     style={
                                            #         "position": "fixed",
                                            #         "height": "15%",
                                            #         "width": "24.5%",
                                            #         "margin-left": "32%",
                                            #         "margin-top": "20px",
                                            #         "box-shadow": "5px 5px 5px  #cacfd2",
                                            #         "margin-right": "5%",
                                            #     },
                                            # ),
                                            html.Div(id="output-state")
                                        ],
                                        type="grow",
                                        color="info",
                                        fullscreen=True,
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    dbc.Spinner(
                                        [
                                            dbc.Card(
                                                [
                                                    dbc.CardHeader(
                                                        "🔤Preprocessed Tokens :"
                                                    ),
                                                    dbc.CardBody(
                                                        id="token",
                                                        style={
                                                            "height": "90px",
                                                            "overflow-y": "scroll",
                                                            "border-radius": "8px",
                                                        },
                                                    ),
                                                ],
                                                color="warning",
                                                outline=True,
                                            ),
                                        ],
                                        type="grow",
                                        color="info",
                                        fullscreen=True,
                                    )
                                ],
                                style={
                                    "margin-top": "10px",
                                    "width": "75%",
                                    "box-shadow": "5px 5px 5px  #cacfd2",
                                },
                            ),
                        ],
                        style={
                            "margin-left": "12.5%",
                            "margin-right": "5%",
                            "width": "auto",
                        },
                    ),
                    width={"size": 6, "order": "second"},
                ),
            ],
            # justify="around",
        ),
    ],
    # style={"background-color": "#e8e7e7"},
)


@app.callback(Output("drop-down", "value"), [Input("radio-button", "value")])
def display_status(selector):
    if selector == "all":
        return [
            "remove_digits",
            "lemmatize_the_text",
            "stop_words",
            "accented_char_removal",
        ]
    else:
        return []


@app.callback(
    [Output("output-state", "children"), Output("token", "children")],
    [Input("submit-button", "n_clicks")],
    [
        State("input_text", "value"),
        State("slider", "value"),
        State("drop-down", "value"),
        State("slider-2", "value"),
    ],
    prevent_initial_call=True,
)
def label_prediction(num_clicks, text, threshold_value, preprocess_func, label_value):
    if text is None or num_clicks is None:
        raise PreventUpdate
    else:
        # list_params = ["remove_digits", "remove_stopwords", "text_lemmatization"]
        # dict_params = [param in preprocess_func for param in params]
        dict_params = {param: True for param in preprocess_func}
        preprocess_text = preprocess(text, **dict_params)
        transformed_text = tfidf.fit_transform([preprocess_text])
        prediction = classifier.predict_proba(transformed_text)
        result = get_tags(prediction[0], threshold_value, label_value)
        final_result = ", ".join(e for e in result)
        tokens = preprocess_text.split(" ")
        final_tokens = ", ".join(f for f in tokens)
        # print(preprocess_text)
        # print(final_result)
        if len(result) < 1:
            return (
                dbc.Alert(
                    dcc.Markdown(
                        "**😔 Looks like I can't find any Tags...Try decreasing the Threshold value**"
                    ),
                    color="warning",
                    duration=120000,
                    style={
                        "position": "fixed",
                        "height": "15%",
                        "width": "24.5%",
                        "margin-left": "32%",
                        "margin-top": "20px",
                        "box-shadow": "5px 5px 5px  #cacfd2",
                        "margin-right": "5%",
                    },
                ),
                final_tokens,
            )
        else:
            return (
                dbc.Alert(
                    [html.H6("Predicted Tags:"), html.Hr(), final_result],
                    color="success",
                    duration=120000,
                    style={
                        "position": "fixed",
                        "height": "15%",
                        "width": "24.5%",
                        "margin-left": "32%",
                        "margin-top": "20px",
                        "box-shadow": "5px 5px 5px  #cacfd2",
                        "margin-right": "5%",
                        "overflow-y": "scroll",
                    },
                ),
                final_tokens,
            )


if __name__ == "__main__":
    app.run_server(debug=True)
