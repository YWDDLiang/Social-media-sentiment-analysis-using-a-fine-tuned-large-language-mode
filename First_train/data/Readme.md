## Data
<div class="table-wrapper" markdown="block">

|                    |                                                    **Cryptocurrency Data**                                                   |                       **Social Media Data**                      |
|--------------------|:----------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------:|
| **Data Source**    |                       [Alpha Vantage API](https://www.alphavantage.co/documentation/#digital-currency)                       | [Snscrape API](https://github.com/JustAnotherArchivist/snscrape) |
| **Queried Data**   |   [meritcircle_user_sentiment](https://github.com/YWDDLiang/llm_research/tree/main/data/preprocess)                          |                                 -                                |
| **Processed Data** | [meritcircle_cleaned_discord](https://github.com/YWDDLiang/llm_research/tree/main/data/preprocess)                           |                                 -                                |

</div>

## Data dictionary
| Variable        | Definition                                          | Description                                         | Frequency     | Range                | Unit        | Type      | Sample Observation                                     |
|-----------------|-----------------------------------------------------|-----------------------------------------------------|---------------|----------------------|-------------|-----------|--------------------------------------------------------|
| AuthorID        | ID of the author                                    | Author's ID                                         | Daily         | Various              | N/A         | String    |3.035150e + 17                                          |
| Author          | Name of the author                                  | Author's name                                       | Daily         | Various              | N/A         | String    |filthyfawkes                                            |
| Data_original   | Time of the data                                    | When this data is sent by author, exact time        | Daily         | 2022/2/14-2023/8/15  | N/A         | String    |2022/2/14 9:12                                          |
| Date            | Date of the data                                    | When this data is sent by author                    | Daily         | 2022/2/14-2023/8/15  | N/A         | String    |2022/2/14                                               |
| Content         | Contents of the data                                | What this author said                               | Daily         | Various              | N/A         | String    |@Professor LP                                           |
| Attachments     | Attachments in the data                             | File or picture attached with the data              | Daily         | Various              | N/A         | String    |                                                        |
| Reactions       | Reaction to the data                                | Other users' reaction to this data                  | Daily         | Various              | N/A         | String    |                                                        |
| Preprocessed    | Preprocessed data                                   | Preprocessed data                                   | Daily         | Various              | N/A         | String    |Professor LP                                            |

</div>

#### Data Source: 
[Alpha Vantage: Digital & Crypto Currencies/DIGITAL_CURRENCY_DAILY](https://www.alphavantage.co/documentation/#digital-currency)
