# K-means-Clustering-on-Text-Documents
Using Scikit-learn, machine learning library for the Python programming language.

Note:
1. Each row in excel sheet corresponds to a document.
2. Data needs to be in excel format for this code, if you have a csv file then you can use **pd.read_csv('file name')** instead of **pd.read_excel('')**. If you don't have any data then just use the dummy corpus given in the code.
3. Clustering is an unsupervised learning technique, which means by using this code you will cluster the set of documents on the basis of some similarity they possess.
4. Note that in 'data.xlsx' the Idea column have the corresponding label/Topic as NA. By applying K-Means you can group similar doc and then label later by applying topic modelling on the groups you have just found out. 

I will be updating topic modelling later. 
