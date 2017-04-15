# Code_3

    dataframe.get_value(index, col) → This is an interesting in that it allows you to 
    search for a specific value based on an index and a column, almost like chess. 
    
    str.split(" "), x.split("") -> Basically splits a string into multiple words. This
    is definitely a good function to remember.
    
    
    
    Featurising a date -> It migth be worth your effort to cut the date into, years 
    (and for some other data, months and days)
    ##
    To do this first do a pd.to_datetime: df["date"] = pd.to_datetime(df["date"]) then
    you separately create the year = df["date"].dt.year, 2007, 2008
    
    Yes there is way to calculate and tabulate all the projections, to weigh them up
    against eachoter. Just look at the way csv's are submitted on kaggle.
    
    Extract selected columns out of pandas dataframe that meets the criteria of 
    belonging to a certain datatype, creare a new dataframe by appending 
    labelencoded columns together:
    
    columns = ["sex", "age_bracket", "blue"]
    for c in columns:
        if df[c].dtype=="object":
            lb = LabelEncoder()
            lb.fit(list(df[c].values)) # Interestin that it is pulling values
            df_main[c] = lb.transform(df[c].values) # Interesting, writes to main df
            features.append(f)
    
    In the data exploration stage it may be worth flipping the features to the verticle
    by transposing a head method, df.head().T
    
    To get dtype, datatype information you can do one of the following:
      df.info()
      df.dtypes
      
    Intead of using that bar chart to calculate the counts you can use a function:
    df["class_feature"].value_counts() 
    
    To winsorise the 99th percentile you can do the following:
    
    ptl = np.percentile(df.feature.values,99),
    
    you see the importance here is the .value that turns it into a percentile 
    importantly, this only gives you the percentile value the next part is to 
    pass it into a dataframe, I like this:
    
    df["feature"] = df["feature"].ix[df["feature"]>ptl] 
    
    Very interesting consignment function: 
    train['num_features'].ix[train['num_features'] > 16] = 16
    which in laymans terms means that every item higher than 16 will equal 16. 
    
    Recall you can passs multiple conditional statements within a dataframe:
    df = df[(df.blue>30) & (df.blue<10)]
    
    Don't ever be afraid to do some manual feature (x) changes such as what is done 
    here:
        x = s.replace("-", "")
        x = x.replace(" ", "")
        x = x.replace("twenty four hour", "24")
        x = x.replace("24/7", "24")
        x = x.replace("24hr", "24")
        x = x.replace("24-hour", "24")
        x = x.replace("24hour", "24")
    
    Examples of one entry:
    df.feature.iloc[1]
    
    You can pass functions into df.columns by adding the function in a mapping method.
    
    df["columns"] = df["columns"].map(function) 
    
    As an example here is a quick process to remove punctuation and the followup mapper.
    
    def removePunctuation(x):
        # Lowercasing all words
        x = x.lower()
        # Removing non ASCII chars
        x = re.sub(r'[^\x00-\x7f]',r' ',x)
        # Removing (replacing with empty spaces actually) all the punctuations
        return re.sub("["+string.punctuation+"]", " ", x)
        
    
    
        
        
        
        
    
    







