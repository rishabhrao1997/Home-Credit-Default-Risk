# Home-Credit-Default-Risk

This is an end to end Machine Learning Case Study, which focusses on building a predictive model by leveraging the dataset provide by Home Credit Group for identifying Potential Loan Defaulters. <br>
The Repository contains following files:<br>
<ol><li><b>EDA - Home Credit Default.ipynb:</b><br>
  This ipynb contains the in-depth EDA for the given dataset. Kindly note that some of the plots might not be visible in the github page (plotly plots), which can be viewed by opening the notebook using nbviewer.</li>
  <li><b>Feature Engineering and Modelling.ipynb</b><br>
    This notebook contains the detailed Feture Engineering and Modelling on the given dataset.</li>
  <li><b>Final.ipynb:</b><br>
    This notebook contains the final pipeline, where the we can directly get the Predictions by just giving the inputs to the pipeline, which does all the pre-processing and predictions by itself.</li>
  <li><b>Deployment Model Trainnig - 300 Features.ipynb</b><br>
    This notebook contains training a model on reduced feature set to reduce the computations requirements for the Deployed Model. This is done keeping in mind the configuration of the AWS EC2 micro instance. </li>
  <li><b>Deployment Folder</b><br>
    This folder contains all the necessary files which would be needed for deploying the web-app on any remote server. Due to file size limitation, the Database is missing from this folder, which can be downloaded from <a href = 'https://drive.google.com/file/d/1nskYRC0xn68qgxeN_Zlz3wqkkulpLlaD/view?usp=sharing'>here</a>, and pasted to inside this 'Final Pipeline Files' folder inside this folder.<br>
  The deployed model can be tested from the link: http://ec2-18-222-96-92.us-east-2.compute.amazonaws.com:5000/</li>  </ol>
