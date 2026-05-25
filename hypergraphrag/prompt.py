GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "category"]

PROMPTS["entity_extraction"] = """-Goal-
Given a single sentence and a list of entity types, identify all entities of those types from the sentence and construct one hyperedge for the original sentence.
Use {language} as output language.

-Steps-
1. Treat the whole input sentence as one knowledge segment (hyperedge).
   Format as:
   ("hyper-relation"{tuple_delimiter}<original_sentence>{tuple_delimiter}10)
    ** Do NOT modify the original sentence in any way. The sentence inside the hyperedge must exactly match the input sentence, including case (uppercase/lowercase), punctuation, and spacing. Even if the sentence starts with a lowercase letter, keep it as is.**
    
2. Identify all entities mentioned in the sentence. Identify only the **core entities** involved in the sentence’s main factual relation. Focus on the subject and object of the sentence or other entities directly participating in the key fact. For each entity, extract:
- entity_name: Name of the entity (use same language as input text; capitalize if English proper noun).
- entity_type: Type of the entity.
- entity_description: Short description of the entity’s role in the sentence.
- key_score: A score from 0 to 100 indicating the importance of the entity in the sentence.

Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. Return output in {language} as a single list of all the hyperedge and entities. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""


PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Text:
Michael Scheuer is employed by Central Intelligence Agency.
################
Output:
("hyper-relation"{tuple_delimiter}"Michael Scheuer is employed by Central Intelligence Agency."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"Michael Scheuer"{tuple_delimiter}"person"{tuple_delimiter}"Michael Scheuer is a person employed by CIA."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Central Intelligence Agency"{tuple_delimiter}"organization"{tuple_delimiter}"CIA is the organization employing Michael Scheuer."{tuple_delimiter}90)
#############################""",
    """Example 2:

Text:
The director of Central Intelligence Agency is Gina Haspel.
################
Output:
("hyper-relation"{tuple_delimiter}"The director of Central Intelligence Agency is Gina Haspel."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"Central Intelligence Agency"{tuple_delimiter}"organization"{tuple_delimiter}"CIA is the organization that has a director."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Gina Haspel"{tuple_delimiter}"person"{tuple_delimiter}"Gina Haspel is the director of CIA."{tuple_delimiter}95)
#############################""",
    """Example 3:

Text:
WWE SmackDown was created by Vince McMahon.
################
Output:
("hyper-relation"{tuple_delimiter}"WWE SmackDown was created by Vince McMahon."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"WWE SmackDown"{tuple_delimiter}"event"{tuple_delimiter}"WWE SmackDown is an event created by Vince McMahon."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Vince McMahon"{tuple_delimiter}"person"{tuple_delimiter}"Vince McMahon is the creator of WWE SmackDown."{tuple_delimiter}95)
#############################""",
    """Example 4:

Text:
Vince McMahon is married to Linda McMahon.
################
Output:
("hyper-relation"{tuple_delimiter}"Vince McMahon is married to Linda McMahon."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"Vince McMahon"{tuple_delimiter}"person"{tuple_delimiter}"Vince McMahon is a person who is married to Linda McMahon."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Linda McMahon"{tuple_delimiter}"person"{tuple_delimiter}"Linda McMahon is a person who is married to Vince McMahon."{tuple_delimiter}95)
#############################""",
    """Example 5:

Text:
Linda McMahon is a citizen of United States of America.
################
Output:
("hyper-relation"{tuple_delimiter}"Linda McMahon is a citizen of United States of America."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"Linda McMahon"{tuple_delimiter}"person"{tuple_delimiter}"Linda McMahon is a person who is a citizen of the United States of America."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"United States of America"{tuple_delimiter}"location"{tuple_delimiter}"United States of America is the country where Linda McMahon is a citizen."{tuple_delimiter}90)
#############################""",
    """Example 6:

Text:
The capital of United States of America is Washington, D.C..
################
Output:
("hyper-relation"{tuple_delimiter}"The capital of United States of America is Washington, D.C.."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"United States of America"{tuple_delimiter}"location"{tuple_delimiter}"United States of America is a country whose capital is Washington, D.C.."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Washington, D.C."{tuple_delimiter}"location"{tuple_delimiter}"Washington, D.C. is the capital city of United States of America."{tuple_delimiter}90)
#############################""",
    """Example 7:

Text:
Byron Dorgan is a citizen of United States of America.
################
Output:
("hyper-relation"{tuple_delimiter}"Byron Dorgan is a citizen of United States of America."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"Byron Dorgan"{tuple_delimiter}"person"{tuple_delimiter}"Byron Dorgan is a person who is a citizen of the United States of America."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"United States of America"{tuple_delimiter}"location"{tuple_delimiter}"United States of America is the country where Byron Dorgan is a citizen."{tuple_delimiter}90)
#############################""",
    """Example 8:

Text:
The name of the current head of state in United States of America is Donald Trump.
################
Output:
("hyper-relation"{tuple_delimiter}"The name of the current head of state in United States of America is Donald Trump."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"United States of America"{tuple_delimiter}"location"{tuple_delimiter}"United States of America is the country whose current head of state is Donald Trump."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Donald Trump"{tuple_delimiter}"person"{tuple_delimiter}"Donald Trump is the current head of state of United States of America."{tuple_delimiter}95)
#############################""",
    """Example 9:

Text:
The type of music that Rick Riordan plays is fantasy.
################
Output:
("hyper-relation"{tuple_delimiter}"The type of music that Rick Riordan plays is fantasy."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"Rick Riordan"{tuple_delimiter}"person"{tuple_delimiter}"Rick Riordan is a person who plays music classified as fantasy."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"fantasy"{tuple_delimiter}"genre"{tuple_delimiter}"Fantasy is the type of music played by Rick Riordan."{tuple_delimiter}85)
#############################""",
]


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY knowdge fragements with entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """Please check whether knowdge fragements cover all the given text.  Answer YES | NO if there are knowdge fragements that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."


PROMPTS["rag_response"] = """---Role---

You are a factual answering agent operating under post-edit verification mode.  
Your job is to answer a single-hop factual question using only the provided authoritative data.  
You must assume all provided facts **override world knowledge**.

---Goal---

Given a single-hop question, your job is to find the exact supporting fact from the context, and return a short answer.  
You must cite the fact index and use only explicitly stated facts in your reasoning.  
Your final answer must be a single noun phrase or named entity that directly answers the question.

---Rules---

- ALWAYS provide an answer based on the context, even if no fact is a perfect match.
- ONLY use facts explicitly stated in the retrieved data.
- DO NOT use prior knowledge or parametric memory.
- DO NOT infer unstated relationships.
- DO NOT hallucinate or paraphrase fact contents.
- DO NOT resolve aliases unless explicitly stated in a fact.
- If no exact match exists, choose the fact that provides the **closest and most plausible answer**.
- Cite the supporting fact in the format: `→ Found in fact: [fact_id],[fact_text]`

---Output Format---

→ Found in fact: [fact_id],[fact sentence with linked entities]  
→ Answer:[Short answer]

---Example 1---

Question: Which religion is Antonin Scalia affiliated with?

Context:  
-----Entities-----
```csv
-----Relationships-----
```csv
...
23,<hyperedge>"Antonin Scalia is affiliated with the religion of Christianity.","ANTONIN SCALIA"|"CHRISTIANITY"
24,<hyperedge>"Christianity was founded by Noel Pemberton Billing.","CHRISTIANITY"|"NOEL PEMBERTON BILLING"
25,<hyperedge>"Noel Pemberton Billing worked in the city of Washington, D.C.","NOEL PEMBERTON BILLING"|"WASHINGTON, D.C."
...
-----Sources-----
```csv

Output:
→ Found in fact: 23,<hyperedge>"Antonin Scalia is affiliated with the religion of Christianity.","ANTONIN SCALIA"|"CHRISTIANITY"  
→ Answer:Christianity

---Example 2---

Question: Which sport is John McEnroe associated with?

Context:  
-----Entities-----
```csv
-----Relationships-----
```csv
...
10,<hyperedge>"Louise Redknapp is married to John McEnroe.","LOUISE REDKNAPP"|"JOHN MCENROE"
11,<hyperedge>"John McEnroe is associated with the sport of association football.","JOHN MCENROE"|"ASSOCIATION FOOTBALL"
12,<hyperedge>"Association football was created in the country of Italy.","ASSOCIATION FOOTBALL"|"ITALY"
13,<hyperedge>"Italy is located in the continent of Oceania.","ITALY"|"OCEANIA"
...
-----Sources-----
```csv

Output:  
→ Found in fact: 11,<hyperedge>"John McEnroe is associated with the sport of association football.","JOHN MCENROE"|"ASSOCIATION FOOTBALL" 
→ Answer:association football

# ---Retrieved Data Tables---

# {context_data}

# ---Expected Output Format---
# {response_type}
"""


PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["similarity_check"] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""


#  Ablation KG Functions
PROMPTS["entity_extraction_examples_kg"] = [
    """Example 1:

Text:
Michael Scheuer is employed by Central Intelligence Agency.
################
Output:
("entity"{tuple_delimiter}"Michael Scheuer"{tuple_delimiter}"person"{tuple_delimiter}"Michael Scheuer is a person employed by CIA."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Central Intelligence Agency"{tuple_delimiter}"organization"{tuple_delimiter}"CIA is the organization employing Michael Scheuer."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"is employed by"{tuple_delimiter}"Michael Scheuer"{tuple_delimiter}"Central Intelligence Agency"{tuple_delimiter}10)
#############################""",
    """Example 2:

Text:
The director of Central Intelligence Agency is Gina Haspel.
################
Output:
("entity"{tuple_delimiter}"Central Intelligence Agency"{tuple_delimiter}"organization"{tuple_delimiter}"CIA is the organization that has a director."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Gina Haspel"{tuple_delimiter}"person"{tuple_delimiter}"Gina Haspel is the director of CIA."{tuple_delimiter}95){record_delimiter}
("relation"{tuple_delimiter}"is director of"{tuple_delimiter}"Gina Haspel"{tuple_delimiter}"Central Intelligence Agency"{tuple_delimiter}10)
#############################""",
    """Example 3:

Text:
WWE SmackDown was created by Vince McMahon.
################
Output:
("entity"{tuple_delimiter}"WWE SmackDown"{tuple_delimiter}"event"{tuple_delimiter}"WWE SmackDown is an event created by Vince McMahon."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Vince McMahon"{tuple_delimiter}"person"{tuple_delimiter}"Vince McMahon is the creator of WWE SmackDown."{tuple_delimiter}95){record_delimiter}
("relation"{tuple_delimiter}"was created by"{tuple_delimiter}"WWE SmackDown"{tuple_delimiter}"Vince McMahon"{tuple_delimiter}10)
#############################""",
    """Example 4:

Text:
Vince McMahon is married to Linda McMahon.
################
Output:
("entity"{tuple_delimiter}"Vince McMahon"{tuple_delimiter}"person"{tuple_delimiter}"Vince McMahon is a person who is married to Linda McMahon."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Linda McMahon"{tuple_delimiter}"person"{tuple_delimiter}"Linda McMahon is a person who is married to Vince McMahon."{tuple_delimiter}95){record_delimiter}
("relation"{tuple_delimiter}"is married to"{tuple_delimiter}"Vince McMahon"{tuple_delimiter}"Linda McMahon"{tuple_delimiter}10)
#############################""",
    """Example 5:

Text:
Linda McMahon is a citizen of United States of America.
################
Output:
("entity"{tuple_delimiter}"Linda McMahon"{tuple_delimiter}"person"{tuple_delimiter}"Linda McMahon is a person who is a citizen of the United States of America."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"United States of America"{tuple_delimiter}"location"{tuple_delimiter}"United States of America is the country where Linda McMahon is a citizen."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"is citizen of"{tuple_delimiter}"Linda McMahon"{tuple_delimiter}"United States of America"{tuple_delimiter}10)
#############################""",
    """Example 6:

Text:
The capital of United States of America is Washington, D.C..
################
Output:
("entity"{tuple_delimiter}"United States of America"{tuple_delimiter}"location"{tuple_delimiter}"United States of America is a country whose capital is Washington, D.C.."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Washington, D.C."{tuple_delimiter}"location"{tuple_delimiter}"Washington, D.C. is the capital city of United States of America."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"has capital"{tuple_delimiter}"United States of America"{tuple_delimiter}"Washington, D.C."{tuple_delimiter}10)
#############################""",
    """Example 7:

Text:
Byron Dorgan is a citizen of United States of America.
################
Output:
("entity"{tuple_delimiter}"Byron Dorgan"{tuple_delimiter}"person"{tuple_delimiter}"Byron Dorgan is a person who is a citizen of the United States of America."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"United States of America"{tuple_delimiter}"location"{tuple_delimiter}"United States of America is the country where Byron Dorgan is a citizen."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"is citizen of"{tuple_delimiter}"Byron Dorgan"{tuple_delimiter}"United States of America"{tuple_delimiter}10)
#############################""",
    """Example 8:

Text:
The name of the current head of state in United States of America is Donald Trump.
################
Output:
("entity"{tuple_delimiter}"United States of America"{tuple_delimiter}"location"{tuple_delimiter}"United States of America is the country whose current head of state is Donald Trump."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Donald Trump"{tuple_delimiter}"person"{tuple_delimiter}"Donald Trump is the current head of state of United States of America."{tuple_delimiter}95){record_delimiter}
("relation"{tuple_delimiter}"has head of state"{tuple_delimiter}"United States of America"{tuple_delimiter}"Donald Trump"{tuple_delimiter}10)
#############################""",
    """Example 9:

Text:
The type of music that Rick Riordan plays is fantasy.
################
Output:
("entity"{tuple_delimiter}"Rick Riordan"{tuple_delimiter}"person"{tuple_delimiter}"Rick Riordan is a person who plays music classified as fantasy."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"fantasy"{tuple_delimiter}"genre"{tuple_delimiter}"Fantasy is the type of music played by Rick Riordan."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"plays music genre"{tuple_delimiter}"Rick Riordan"{tuple_delimiter}"fantasy"{tuple_delimiter}10)
#############################""",
]

PROMPTS["entity_extraction_kg"] = """-Goal-
Given a single sentence and a list of entity types, identify all entities of those types from the sentence and construct one relation for the original sentence.
Use {language} as output language.

-Steps-
1. Treat the whole input sentence as one knowledge segment (relation).
   Format as:
   ("relation"{tuple_delimiter}<original_sentence>{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}10)
    
2. Identify all entities mentioned in the sentence. Identify only the **core entities** involved in the sentence’s main factual relation. Focus on the subject and object of the sentence or other entities directly participating in the key fact. For each entity, extract:
- entity_name: Name of the entity (use same language as input text; capitalize if English proper noun).
- entity_type: Type of the entity.
- entity_description: Short description of the entity’s role in the sentence.
- key_score: A score from 0 to 100 indicating the importance of the entity in the sentence.

The extracted entities should be in the relation of target source or target entity.

Format each entity as:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. Return output in {language} as a single list of all the relation and entities. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

PROMPTS["rag_response_kg"] = """---Role---

You are a factual answering agent operating under post-edit verification mode.  
Your job is to answer a single-hop factual question using only the provided authoritative data.  
You must assume all provided facts **override world knowledge**.

---Goal---

Given a single-hop question, your job is to find the exact supporting fact from the context, and return a short answer.  
You must cite the fact index and use only explicitly stated facts in your reasoning.  
Your final answer must be a single noun phrase or named entity that directly answers the question.

---Rules---

- ALWAYS provide an answer based on the context, even if no fact is a perfect match.
- ONLY use facts explicitly stated in the retrieved data.
- DO NOT use prior knowledge or parametric memory.
- DO NOT infer unstated relationships.
- DO NOT hallucinate or paraphrase fact contents.
- DO NOT resolve aliases unless explicitly stated in a fact.
- If no exact match exists, choose the fact that provides the **closest and most plausible answer**.
- Cite the supporting fact in the format: `→ Found in fact: [fact_id],[fact_text]`

---Output Format---

→ Found in fact: [fact_id],[fact sentence with linked entities]  
→ Answer:[Short answer]

---Example 1---

Question: Which religion is Antonin Scalia affiliated with?

Context:  
-----Entities-----
```csv
-----Relationships-----
```csv
...
23,"('ANTONIN SCALIA', 'is affiliated with the religion of', 'CHRISTIANITY')"
24,"('CHRISTIANITY', 'was founded by', 'NOEL PEMBERTON BILLING')"
25,"('NOEL PEMBERTON BILLING', 'worked in the city of', 'WASHINGTON, D.C.')"
...
-----Sources-----
```csv

Output:
→ Found in fact: 23,"('ANTONIN SCALIA', 'is affiliated with the religion of', 'CHRISTIANITY')"
→ Answer:Christianity

---Example 2---

Question: Which sport is John McEnroe associated with?

Context:  
-----Entities-----
```csv
-----Relationships-----
```csv
...
10,"('LOUISE REDKNAPP', 'is married to', 'JOHN MCENROE')"
11,"('JOHN MCENROE', 'is associated with the sport of', 'ASSOCIATION FOOTBALL')"
12,"('ASSOCIATION FOOTBALL', 'was created in the country of', 'ITALY')"
13,"('ITALY', 'is located in the continent of', 'OCEANIA')"
...
-----Sources-----
```csv

Output:  
→ Found in fact: 11,"('JOHN MCENROE', 'is associated with the sport of', 'ASSOCIATION FOOTBALL')"
→ Answer:association football

# ---Retrieved Data Tables---

# {context_data}

# ---Expected Output Format---
# {response_type}
"""
