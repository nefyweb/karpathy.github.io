# Code 4
I found the perfect package for XGBOOST in py36:


    conda install -c conda-forge xgboost=0.6a2

You can train on a Target and test on another. At eHarmony they trained on profile view but tested on 7 day communication as predictor of affinity and found much better accuracy.

The amount of data is so much better, the dataset is only as big as the smallest class. 

Change external, deterministic.

Interesting, not sure why but people do this:


    #Shuffle the dataframes so that the training is done in a random order.
    X_train = shuffle(X_train)
    X_test = shuffle(X_test)


    param = {}
    
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    
    Quite interesting, you can have the above and it bascially does not create a dataframe but rather a dictionary, but the names are still on the left. The key is then kept in the [] brackets and the value on the right. I did not quite know about it. 
    
    Training the XGBMatrix is rather easy:
    1. First feed it into a matrix that XGB understands, in this case DMatrix 
    # DMatrix is a internal data structure that used by XGBoost which is optimized for both memory efficiency and training speed



    XGB:
    
    Assume three binary targets, like_hats, like_gardening, like_gaming.
     
    -> We end up making the following assumptions:
    We can use age as a reaonable predictor. Old people: Likes gardening, Doesn't like games, ambigous towards hats.
    
                                              Young people: Likes games, dislikes gardning, ambigious towards hats. 
    
    You create a regression tree and make sure that on the last split there is at least three samples at the terminal node. 



https://d2mxuefqeaa7sj.cloudfront.net/s_9B1ABDCCE57F5E27C3D71E42B48793BDB8D974D1E82B41B932143E43561AA7CA_1492900773635_image.png


Here it overfits, because the terminal tree has to have three nodes. In the above tree the assumption was also made that the gardening target would be the best opening node. Also it picks up like hats which was normal from the start. 


    Here is the issue with a single regression tree (decision tree) - it does not include the predictive power of multiple overlapping regions of the feature space. (As it only starts with gardening)


    Feature space: Likes Gardening, Likes Hats, Likes Gaming, therefore R3 is the potential feature space. 

Therefore a better tree would be the following: 


https://d2mxuefqeaa7sj.cloudfront.net/s_9B1ABDCCE57F5E27C3D71E42B48793BDB8D974D1E82B41B932143E43561AA7CA_1492901254240_image.png


The issue is that the overfit tree has much smaller regressions, we simply used one variable to calculate the age prediction because we do not want to overfit. When you do not overfit there might be a requirement to introduce more variables for better prediction accuracy. 


https://d2mxuefqeaa7sj.cloudfront.net/s_9B1ABDCCE57F5E27C3D71E42B48793BDB8D974D1E82B41B932143E43561AA7CA_1492901397272_image.png



    Above is an example of a prediciton and residual table based on the estimates made. The errors are quite significant. What we then can do is fit a second regression tree on the risiduals of the first tree and introduce a new variables, which can now better sort out and predict residuals. 



https://d2mxuefqeaa7sj.cloudfront.net/s_9B1ABDCCE57F5E27C3D71E42B48793BDB8D974D1E82B41B932143E43561AA7CA_1492901508338_image.png



    Above is an example of a prediciton and residual table based on the estimates made. The errors are quite significant. What we then can do is fit a second regression tree on the risiduals of the first tree.

Now instead of digging into a smaller sample (nodes with less samples) to predict items such as like_hats (as can be seen in the overfit tree) with a small region of the input space. The new approach allows random noise to select like_hats as a splitting feature. 



    We can then inprove the predictions of the first tree by adding the "correcting" predictions from this tree


| **PersonID** | **Age** | **Tree1 Prediction** | **Tree1 Residual** | **Tree2 Prediction/**
**Residual** | **Combined Prediction** | **Final Residual** |
| ------------ | ------- | -------------------- | ------------------ | ---------------------------------- | ----------------------- | ------------------ |
| 1            | 13      | 19.25                | -6.25              | -3.567                             | 15.68                   | 2.683              |
| 2            | 14      | 19.25                | -5.25              | -3.567                             | 15.68                   | 1.683              |
| 3            | 15      | 19.25                | -4.25              | -3.567                             | 15.68                   | 0.6833             |
| 4            | 25      | 57.2                 | -32.2              | -3.567                             | 53.63                   | 28.63              |
| 5            | 35      | 19.25                | 15.75              | -3.567                             | 15.68                   | -19.32             |
| 6            | 49      | 57.2                 | -8.2               | 7.133                              | 64.33                   | 15.33              |
| 7            | 68      | 57.2                 | 10.8               | -3.567                             | 53.63                   | -14.37             |
| 8            | 71      | 57.2                 | 13.8               | 7.133                              | 64.33                   | -6.667             |
| 9            | 73      | 57.2                 | 15.8               | 7.133                              | 64.33                   | -8.667             |

Tree 1 Residual + Tree 2 Prediction = Intermediate Residual; 
Tree 1 Prediction - Intermediate Residual = Tree Level 2 Predcition;
Tree Level 2 Prediction - Actual Age = Tree Level 2 Residual .

| **Tree1 SSE** | **Combined SSE** |
| ------------- | ---------------- |
| 1994          | 1765             |


As you can see above the sum of squared errors is much lower come the second tree. 

So above is a Naive formulation of gradient boosting. 


    Fp(), Fr() = Regression Model
    x = Feature Inputs
    y = Target Value
    Residuals(x) = (y1 -Fp(x))
    Fp2() = Fr[(y - Fp(x))] + Fp(x)
    Fp3() = Fr2[y - (Fp2(x) )]  + Fp2(x)
    Fp3() = Fr2[y - (Fr[(y1 - Fp(x))] + Fp(x) )] + Fr[(y1 - Fp(x))] + Fp(x)
    
    Simple way of writing:
    
    Fpm+1(x) = Fpm(x) + Frm(x) = y, for m >0
    
    You can insert more and more models untill the errors are perfected:
    
    Now we can conform to most gradient boosting implementiation - we will initialise the model with a single prediction value. 
    
    The task now is to minimise the square error: 
    1. We will start off my initialising F with the means of the training targer vales.
    2. And then the question is one of how should we adjust m, i.e. how many times should we iterate the resdiual-correction procedure untill we decide upon the final model, F. 
    3. The best approach here is to test various lvalues of m via cross validation. 



    Instead of minimising sqaured errors, we can also go ahead and minimise anasalute errors.
    
    http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/
    
    Much more to this but bored now. 


.highlighter-rouge, .highlighter-rouge code, .highlighter-rouge pre  { background: #29281e; }


| <link rel="stylesheet" href="{{ "/css/main.css" | prepend: site.baseurl }}"> |





