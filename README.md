# GenAI Marketing AI Assistant

This is a prototype application, so the current deployment is not fully scalable. At this stage, anyone with an access link can use the app. However, please note that the algorithm has specific formal requirements for the input data. 

To simplify testing, the script **RAG.py** generates a randomized dataframe and automatically stores it as a `.csv` file named **RAG.csv**. This output file meets the algorithm's requirements and can be used to explore the app's functionalities.

### Data Requirements

The following columns are required in any `.csv` file used with this prototype:

- **Name**: Name of the customer or entity.
- **SalesAmount**: Numeric value representing the sales amount. - must be numeric
- **ItemBought**: Description of the purchased item.
- **PurchaseDate**: Date of purchase. - must be datetime
- **Country**: Country of the customer or purchase.
- **Review**: Review text provided by the customer.
- **ExperienceRating**: Numerical rating of the customer experience.
- **Comment**: Additional comments or feedback.
- **Latitude**: Latitude of the customer's location. - not necessary
- **Longitude**: Longitude of the customer's location. - not necessary


While this prototype is tailored to the above data structure, the model can be adjusted to accommodate different formats and requirements as needed.

### How to run locally

Pull this repository into any Python 3.10 IDE. In the directory where the script **HOME.py** is featured, run `streamlit run HOME.py` in the terminal. A web page will open on your default browser with the application running.
Make sure to have the files `config.toml` (contains theme specifications) and `.env` (contains the API key) in the correct directory. 
