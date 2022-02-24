##############################################################
# 1 : Imports
##############################################################

##############################################################
# 1.1 : FastAPI
##############################################################

import uvicorn

from fastapi import FastAPI
from fastapi import Header
from fastapi import HTTPException
from fastapi import status
from fastapi import Depends

from fastapi.security import HTTPBasic
from fastapi.security import HTTPBasicCredentials

from pydantic import BaseModel
from typing import Optional

##############################################################
# 1.2 : Model
##############################################################

##############################################################
# 1.3 : Others
##############################################################

import os

##############################################################
# 2 : Initialisations
##############################################################

##############################################################
# 2.1 : Credentials base
##############################################################

users_base = {
  "alice": "wonderland",
  "bob": "builder",
  "clementine": "mandarine"
}

admin_base = {
  "admin": os.environ.get('ADMIN_PWD')
}

##############################################################
# 2.2 : Query and responses definitions
##############################################################

# model score query and response structures
# also model initialization response parameters

class ModelScoreQuery(BaseModel) :
    branch : Optional[str] = 'Compound'

class ModelScoreResponse(BaseModel):
    score : float
    branch : str = 'Compound'

# sentiment predict query and response structures

class SentimentQuery(BaseModel) :
    sentence : str
    branch : Optional[str] = 'Compound'

class SentimentResponse(BaseModel):
    sentiment : int
    branch : str = 'Compound'

# usual http errors
http_responses = {
                  200: {"description": "OK"}
                  , 401: {"description": "Parameters error"}
                  , 405: {"description": "Request type error"}
                  , 500: {"description": "Predict model error"}
                 }

##############################################################
# 2.2 : Server environment
##############################################################

# creating a FastAPI server
server = FastAPI(title = 'Sentiment Prediction API')

security = HTTPBasic()

##############################################################
# 2.3 : Files environment
##############################################################

folder = 'data/'
csv_file = 'DisneylandReviews.csv'
sentiment_dump_file = 'sentiment.pckl'
count_vectorizer_dump_file = 'count_vectorizer.pckl'
score_dump_file = 'score.pckl'
method1 = 'method_01_'
method2 = 'method_02_'

##############################################################
# 3 : Utilities
##############################################################

##############################################################
# 3.1 : Checking credentials
##############################################################

def is_in_dict(dict, user, password) :

    try:
        return (dict[user] == password)
    except KeyError :
        return False

def check_authentication(authentication, user_or_admin) :

    if authentication == "" :
        return {}

    user = authentication.username
    password = authentication.password

    if (authentication.username == "") or (authentication.password == "") :
        return {}

    # if user, check both user and admin
    if user_or_admin == "user" :
        if is_in_dict(users_base, user, password) :
            return {'0' : user}
        elif is_in_dict(admin_base, user, password) :
            return {'0' : user}
        else :
            return {}
    # if admin, check only admin
    elif user_or_admin == "admin" :
        if is_in_dict(admin_base, user, password) :
            return {'0' : user}
    else :
        return {}

##############################################################
# 3.2 : Processing text
##############################################################

def preprocess_text(text) :

    from nltk.corpus import stopwords
    from nltk.tokenize import NLTKWordTokenizer

    stop_words = set(stopwords.words('english'))
    stop_words.update(["'ve"
                       , ""
                       , "'ll"
                       , "'s"
                       , "."
                       , ","
                       , "?"
                       , "!"
                       , "("
                       , ")"
                       , ".."
                       , "'m"
                       , "'n"
                       , "n"
                       , "u"
                      ])

    tokenizer = NLTKWordTokenizer()
    
    text = text.lower()
    
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    
    return ' '.join(tokens)

###############################################################
# 3.3 : Computing sentiment model with method 1
###############################################################

def compute_sentiment_model_1() :

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    import pandas as pd
    import numpy as np

    from joblib import dump

    # file path
    csv_filename = folder + csv_file

    # reading data file
    df = pd.read_csv(csv_filename, encoding = 'cp1252')

    # preparing data for the model
    df = df.drop(['Review_ID', 'Year_Month', 'Reviewer_Location'], axis = 1)
    df['Review_Text'] = df['Review_Text'].apply(preprocess_text)

    features = df['Review_Text']
    target = df['Rating']

    # training model
    X_train, X_test, y_train, y_test = train_test_split(features, target)

    count_vectorizer = CountVectorizer(max_features = 2000)

    X_train_cv = count_vectorizer.fit_transform(X_train)
    X_test_cv = count_vectorizer.transform(X_test)

    # model = RandomForestClassifier(max_depth = 3, n_estimators = 100)
    model = LogisticRegression()
    # model = DecisionTreeClassifier(max_depth = 8)

    model.fit(X_train_cv, y_train)

    # computing model score
    score = model.score(X_test_cv, y_test)

    # saving model
    sentiment_dump_filename = folder + method1 + sentiment_dump_file
    print(str(model), 'saved at : ', sentiment_dump_file)

    dump(model, sentiment_dump_filename)

    # saving count_vectorizer
    count_vectorizer_dump_filename = folder + method1 + count_vectorizer_dump_file
    print(str(count_vectorizer), 'saved at : ', count_vectorizer_dump_filename)

    dump(count_vectorizer, count_vectorizer_dump_filename)

    # saving score
    score_dump_filename = folder + method1 + score_dump_file
    print(str(score), 'saved at : ', score_dump_filename)

    dump(score, score_dump_filename)

    return  ModelScoreResponse(score = score, branch = 'Compound')

###############################################################
# 3.4 : Computing sentiment model with method 2
###############################################################

def compute_sentiment_model_2() :

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    import pandas as pd
    import numpy as np

    from joblib import dump

    # file path
    csv_filename = folder + csv_file

    # reading data file
    df = pd.read_csv(csv_filename, encoding = 'cp1252')

    # preparing data for the model
    df = df.drop(['Review_ID', 'Year_Month', 'Reviewer_Location'], axis = 1)
    df['Review_Text'] = df['Review_Text'].apply(preprocess_text)

    count_vectorizers = {}
    models = {}
    scores = {}

    nb_branches = df['Branch'].nunique()
    model_score = 0

    for branch in df['Branch'].unique():
        count_vectorizer = CountVectorizer(max_features = 2000)

        # model = LogisticRegression()
        model = RandomForestClassifier(n_estimators = 20, max_depth = 5)

        df_temp = df[df['Branch'] == branch]

        # training model
        X_train, X_test, y_train, y_test = train_test_split(df_temp['Review_Text']
                                                            , df_temp['Rating'])

        X_train_cv = count_vectorizer.fit_transform(X_train)
        X_test_cv = count_vectorizer.transform(X_test)

        model.fit(X_train_cv, y_train)

        score = model.score(X_test_cv, y_test)
        model_score += score

        count_vectorizers[branch] = count_vectorizer
        models[branch] = model
        scores[branch] = score

    model_score /= nb_branches

    # saving models
    sentiment_dump_filename = folder + method2 + sentiment_dump_file
    print(str(models), 'saved at : ', sentiment_dump_file)

    dump(models, sentiment_dump_filename)

    # saving count_vectorizers 
    count_vectorizer_dump_filename = folder + method2 + count_vectorizer_dump_file
    print(str(count_vectorizers), 'saved at : ', count_vectorizer_dump_filename)

    dump(count_vectorizers, count_vectorizer_dump_filename)

    # saving scores
    score_dump_filename = folder + method2 + score_dump_file
    print(str(scores), 'saved at : ', score_dump_filename)

    dump(scores, score_dump_filename)

    return  ModelScoreResponse(score = model_score, branch = 'Compound')

###############################################################
# 3.5 : Predicting sentiment with previous model 1
###############################################################

def predict_sentiment_model_1(sentiment_dict) :

    import pandas as pd
    import numpy as np

    from joblib import load

    # file path
    sentiment_dump_filename = folder + method1 + sentiment_dump_file
    model = load(sentiment_dump_filename)

    # creating the dataframe from the entry as a dictionary
    # it has the same structure as the dataframe read from the csv file
    df = pd.DataFrame({'Review_Text' : sentiment_dict['sentence']
                       , 'Branch' : sentiment_dict['branch']}, index = [0]).reset_index()
    df['Review_Text'] = df['Review_Text'].apply(preprocess_text)

    features = df['Review_Text']
    # row_1 = features.iloc[0]
    # print(row_1)

    # file path
    count_vectorizer_dump_filename = folder + method1 + count_vectorizer_dump_file
    count_vectorizer = load(count_vectorizer_dump_filename)

    X_cv = count_vectorizer.transform(features)

    prediction = pd.DataFrame(model.predict(X_cv))

    sentiment = prediction.values[0]
    branch = sentiment_dict['branch']

    return SentimentResponse(sentiment = sentiment, branch = branch)

###############################################################
# 3.6 : Predicting sentiment with previous model 2
###############################################################

def predict_sentiment_model_2(sentiment_dict) :

    import pandas as pd
    import numpy as np

    from joblib import load

    # file path
    sentiment_dump_filename = folder + method2 + sentiment_dump_file
    models = load(sentiment_dump_filename)

    # creating the dataframe from the entry as a dictionary
    # it has the same structure as the dataframe read from the csv file
    df = pd.DataFrame({'Review_Text' : sentiment_dict['sentence']
                       , 'Branch' : sentiment_dict['branch']}, index = [0]).reset_index()
    df['Review_Text'] = df['Review_Text'].apply(preprocess_text)

    features = df['Review_Text']
    # row_1 = features.iloc[0]
    # print(row_1)

    # file path
    count_vectorizer_dump_filename = folder + method2 + count_vectorizer_dump_file
    count_vectorizers = load(count_vectorizer_dump_filename)
    print(sentiment_dict['branch'])
    X_cv = count_vectorizers[sentiment_dict['branch']].transform(features)

#    prediction = pd.DataFrame(models[sentiment_dict['branch']].predict(X_cv))
    prediction = models[sentiment_dict['branch']].predict(X_cv)
    print(prediction)
#    sentiment = prediction.values[0]
    sentiment = prediction
    branch = sentiment_dict['branch']

    return SentimentResponse(sentiment = sentiment, branch = branch)

###############################################################
# 3.7 : Giving model score with previous model 1
###############################################################

def get_score_model_1(score_dict) :

    import pandas as pd
    import numpy as np

    from joblib import load

    # file path
    score_dump_filename = folder + method1 + score_dump_file
    score = load(score_dump_filename)

    branch = 'Compound'

    return ModelScoreResponse(score = score, branch = branch)

###############################################################
# 3.8 : Giving model score with previous model 2
###############################################################

def get_score_model_2(score_dict) :

    import pandas as pd
    import numpy as np

    from joblib import load

    # file path
    score_dump_filename = folder + method2 + score_dump_file
    scores = load(score_dump_filename)
    print(score_dict['branch'])
    if score_dict['branch'] == 'Compound' :
        print(scores)
        nb_branches = len(scores)
        score = 0

        for key, value in scores.items():
            score += value

        score /= nb_branches
    else : 
        print(score_dict['branch'], '2')
        score = scores[score_dict['branch']]

    branch = score_dict['branch']

    return ModelScoreResponse(score = score, branch = branch)

###############################################################
# 4 : API
###############################################################

##############################################################
# 4.1 : API : check status
##############################################################

@server.get(
            '/sentiment/status'
            , name = "application is functional"
            , responses = http_responses
           )
async def get_status() :
    """
    Response : {'check': 'Ok'}
    """

    return {
            'check': 'Ok'
           }

##############################################################
# 4.2 : API : initialize (train) model 1
##############################################################

@server.post(
             '/sentiment/m1/init'
             , response_model = ModelScoreResponse
             , name = "Sentiment model initialization api for model 1"
             , responses = http_responses
            )
async def post_compute_sentiment_model_1(
                                         credentials : HTTPBasicCredentials = Depends(security)
                                        ) :
    """
    The user sends a request without body to initialise the model 1

    Admin credentials are requested

    The api returns the regression model score
    """

    if not credentials :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Absent username and password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if not check_authentication(credentials, "admin") :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect username or password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    model_score_response = compute_sentiment_model_1()

    return model_score_response

##############################################################
# 4.3 : API : initialize (train) model 2
##############################################################

@server.post(
             '/sentiment/m2/init'
             , response_model = ModelScoreResponse
             , name = "Sentiment model initialization api for model 2"
             , responses = http_responses
            )
async def post_compute_sentiment_model_2(
                                         credentials : HTTPBasicCredentials = Depends(security)
                                        ) :
    """
    The user sends a request without body to initialise the model 2

    Admin credentials are requested

    The api returns the regression model score
    """

    if not credentials :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Absent username and password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if not check_authentication(credentials, "admin") :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect username or password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    model_score_response = compute_sentiment_model_2()

    return model_score_response

##############################################################
# 4.4 : API : predict sentiment for one sentence for model 1
##############################################################

@server.post(
             '/sentiment/m1/predict'
             , response_model = SentimentResponse
             , name = "Sentiment prediction api for model 1"
             , responses = http_responses
            )
async def post_predict_sentiment_model_1(
                                         sentiment_query : SentimentQuery
                                         , credentials : HTTPBasicCredentials = Depends(security)
                                        ) :
    """
    The user sends a sentence and a branch in the request body

    If present, the branch must be 'Compound'

    Credentials are requested

    The api returns the predicted sentiment for model 1
    """

    if not credentials :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Absent username and password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if not check_authentication(credentials, "user") :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect username or password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if sentiment_query.branch != 'Compound' :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect branch'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    sentiment_dict = {
                      'sentence' : sentiment_query.sentence
                      , 'branch' : sentiment_query.branch
                     }

    sentiment_response = predict_sentiment_model_1(sentiment_dict)

    return sentiment_response

##############################################################
# 4.5 : API : predict sentiment for one sentence for model 2
##############################################################

@server.post(
             '/sentiment/m2/predict'
             , response_model = SentimentResponse
             , name = "Sentiment prediction api for model 2"
             , responses = http_responses
            )
async def post_predict_sentiment_model_2(
                                         sentiment_query : SentimentQuery
                                         , credentials : HTTPBasicCredentials = Depends(security)
                                        ) :
    """
    The user sends a sentence and a branch in the request body

    The branch must be in ['Compound', 'Disneyland_HongKong', 'Disneyland_California ', 'Disneyland_Paris']

    Credentials are requested

    The api returns the predicted sentiment for model 2
    """

    if not credentials :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Absent username and password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if not check_authentication(credentials, "user") :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect username or password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if sentiment_query.branch not in [
                                      'Disneyland_HongKong'
                                      , 'Disneyland_California '
                                      , 'Disneyland_Paris'
                                     ] :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect branch'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    sentiment_dict = {
                      'sentence' : sentiment_query.sentence
                      , 'branch' : sentiment_query.branch
                     }

   
    sentiment_response = predict_sentiment_model_2(sentiment_dict)
    return sentiment_response

##############################################################
# 4.6 : API : get score for model 1
##############################################################

@server.post(
             '/sentiment/m1/score'
             , response_model = ModelScoreResponse
             , name = "Model score prediction api for model 1"
             , responses = http_responses
            )
async def post_get_score_model_1(
                                 model_score_query : ModelScoreQuery
                                 , credentials : HTTPBasicCredentials = Depends(security)
                                ) :
    """
    If present in the request body, the branch must be 'Compound'

    Credentials are requested

    The api returns the score for model 1
    """

    if not credentials :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Absent username and password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if not check_authentication(credentials, "user") :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect username or password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if model_score_query.branch != 'Compound' :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect branch'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    score_dict = {
                  'branch' : model_score_query.branch
                 }

    score_response = get_score_model_1(score_dict)

    return score_response

##############################################################
# 4.7 : API : get score for model 2
##############################################################

@server.post(
             '/sentiment/m2/score'
             , response_model = ModelScoreResponse
             , name = "Model score prediction api for model 2"
             , responses = http_responses
            )
async def post_get_score_model_2(
                                 model_score_query : ModelScoreQuery
                                 , credentials : HTTPBasicCredentials = Depends(security)
                                ) :
    """
    The user sends the branch in the request body

    The branch must be in ['Compound', 'Disneyland_HongKong', 'Disneyland_California ', 'Disneyland_Paris']

    Credentials are requested

    The api returns the score for model 2
    """

    if not credentials :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Absent username and password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if not check_authentication(credentials, "user") :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect username or password'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    if model_score_query.branch not in [
                                      'Compound'
                                      , 'Disneyland_HongKong'
                                      , 'Disneyland_California '
                                      , 'Disneyland_Paris'
                                     ] :
        raise HTTPException(
                            status_code = status.HTTP_401_UNAUTHORIZED
                            , detail = 'Incorrect branch'
                            , headers = {"WWW-Authenticate": "Basic"}
                           )

    score_dict = {
                  'branch' : model_score_query.branch
                 }


    score_response = get_score_model_2(score_dict)


    return score_response

##############################################################
# 5 : main
##############################################################

if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=8000)

