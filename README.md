# FlavourNet

A text-to-network engine: it turns Yelp reviews of coffee shops into a
knowledge graph of concepts (drinks, food, service, ambience, quality) and
the relationships customers draw between them, then analyses that network
and lets you explore it in an interactive dashboard.

Built as a team project for a social network analysis unit, with
contributions from Sushant Shrestha and Shrey Patel.

Work in progress. Being added stage by stage:

- [ ] build a 10,000-review Coffee & Tea sample from the Yelp Open Dataset
- [ ] concept extraction (TF-IDF, YAKE, spaCy NER)
- [ ] relationship extraction (dependency parsing) and PMI filtering
- [ ] knowledge graph, centrality, communities and sentiment
- [ ] interactive Dash dashboard
