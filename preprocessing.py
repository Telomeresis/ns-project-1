# Import for scraping
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import re
import pickle
import random

# Import for network and plot creation
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

def preprocessing():

    # Step 1: Loading Provided Faculty Details
    def load_faculty():
        # Import faculty details into df
        faculty_df = pd.read_excel('Faculty.xlsx')
        # Select relevant columns
        faculty_df = faculty_df[['Faculty', 'Position', 'Gender', 'Management', 'DBLP', 'Area']]
        return faculty_df
    faculty_df = load_faculty()
    print('Step 1 Completed\n')

    # Step 2: Scraping DBLP HTML
    def scrape_dblp(faculty_df):

        # Check if content_list.pkl exists.
        try:
            f = open("content_list.pkl")
            with open('content_list.pkl', 'rb') as f:
                content_list = pickle.load(f)
            # Do something with the file
        except IOError:
            f = None

            print("File not accessible")
            # Create empty list for storing modified DBLP link to access XML variant w/ their API
            xml_list = []
            # Iterate over faculty_df and replace .html w/ .xml - updated to append .xml for missing .html cases
            for each in faculty_df['DBLP']:
                if '.html' in each:
                    replaced_each = each.replace(".html", ".xml")
                xml_list.append(replaced_each)

            # Declare list to store extracted content
            content_list = []

            i = 0
            # Iterate using q_list to make a GET request to fetch raw HTML content
            for each in xml_list:
                html_content = requests.get(each).text
                content_list.append(html_content)
                i+=1
                if (i % 10 == 0):
                    print(str(i)+' Records Retrieved')

            # Store content_list with pickle
            with open('content_list.pkl', 'wb') as f:
                pickle.dump(content_list, f)
        finally:
            f.close()

        return content_list
    content_list = scrape_dblp(faculty_df)
    print('Step 2 Completed\n')

    # Step 3: Parsing DBLP HTML w/ BS4
    def parse_dblp(content_list):

        # Declare empty list for storing soups
        pretty_soup_list = []

        for each in content_list:
            soup = BeautifulSoup(each, "lxml")
            pretty_soup_list.append(soup.prettify())

        # Store pretty_soup_list with pickle
        with open('pretty_soup_list.pkl', 'wb') as f:
            pickle.dump(pretty_soup_list, f)

        return pretty_soup_list
    pretty_soup_list = parse_dblp(content_list)
    def parse_articles_conf(pretty_soup_list):
        # Declare empty all_article_list
        all_article_list = []
        faculty_dblp_name_list = []

        # Iterate over pretty_soup_list + extract names because given names in excel aren't same as DBLP lmao
        for each in pretty_soup_list:
            converted_each = BeautifulSoup(each, "lxml") # need to convert lmao
            individual_article_list = converted_each.find_all('article')
            individual_article_list += converted_each.find_all('inproceedings')
            all_article_list.append(individual_article_list)
            try:
                faculty_dblp_name = converted_each.dblpperson['name']
            except:
                faculty_dblp_name = converted_each.title.text.strip().strip('dblp: ') # omg cancerous code sorry
            finally:
                #print(faculty_dblp_name)
                faculty_dblp_name_list.append(faculty_dblp_name)

        return all_article_list, faculty_dblp_name_list
    all_article_list, faculty_dblp_name_list = parse_articles_conf(pretty_soup_list)
    print('Step 3 Completed\n')

    # Step 4.1: Declare DF for DBLP
    COLUMN_NAMES=[
        'f_index',
        'Faculty',
        'key',
        'Year',
        'Full Authors List'
    ]
    dblp_df = pd.DataFrame(columns=COLUMN_NAMES)

    # Step 4.2: DF population w/ Parsed DBLP Data:
    def df_population(dblp_df):
        # Index for doing dict mapping later
        faculty_index = 0

        # declare empty lists for DF
        article_key_list = []
        article_mdate_list = []
        faculty_index_list = []
        title_list = []
        year_list = []
        authors_list = []

        for each in all_article_list:
            for article in each:
                # Article Tag Extraction w/ Array Indexing
                article_key = article["key"]
                article_mdate = article["mdate"]
                # Strip processing
                stripped_year = article.year.text.strip()
                stripped_authors = [each.text.strip() for each in article.find_all('author')] # list comprehension; bad space and time complexity
                # Append to df
                append_dict = {'f_index': faculty_index, 'Faculty': '', 'key': article_key, 'Year': stripped_year, 'Full Authors List': stripped_authors}
                dblp_df = dblp_df.append(append_dict, ignore_index=True)

            faculty_index+=1

        # Create dict mapping for Faculty, len is used as f_index.
        faculty_dict_mapping = dict(zip(range(len(faculty_df['Faculty'])), faculty_df['Faculty'],))
        dblp_df['Faculty'] = dblp_df['f_index'].map(faculty_dict_mapping)

        # Store dblp_df with pickle
        with open('dblp_12k_df.pkl', 'wb') as f:
            pickle.dump(dblp_df, f)
        return dblp_df
    dblp_df = df_population(dblp_df)
    print('Step 4 Completed\n')

    # Step 5: Dataframe Post Processing (NLP & Metrics)
    def find_name_form_in_list(fac_name, authors_list, dblp_df):
        mod_fac_name = fac_name.replace("-", " ")
        mod_fac_name_list = mod_fac_name.split(" ")
        #print(mod_fac_name_list)
        best_i = 0
        best_count = 0

        for i in range(len(authors_list)):
            #check matchability
            author = authors_list[i]
            count = 0
            for w in mod_fac_name_list:
                if w in author:
                    count += 1
            if count > best_count:
                best_count = count
                best_i = i
        return authors_list[best_i]
    dblp_df["published_name"] = dblp_df.apply(lambda row: find_name_form_in_list(row["Faculty"], row["Full Authors List"], dblp_df), axis = 1)
    def ci_calculation(dblp_df):
        # Code for inserting contribution into DF
        contribution_index_list = []
        for i, row in dblp_df.iterrows():
            if (row['published_name'] in row['Full Authors List']): # check if DBLP name exists in Full Authors List
                ci = row['Full Authors List'].index(row['published_name'])+1 # if so, retrieve index, +1 (to acccount for 0), then append to contribution_index_list
            else:
                ci = '-'
            contribution_index_list.append(ci)
        dblp_df['Author Contribution Index'] = contribution_index_list # assigns a value based on how much contribution the author has made for a publication. 1 = Highest (Main)
        return dblp_df
    dblp_df = ci_calculation(dblp_df)
    authors_list = dblp_df['Full Authors List'].tolist()
    dblp_df['Other Authors'] = dblp_df['Full Authors List'].copy() # assigns a value based on how much contribution the author has made for a publication. 1 = Highest (Main)
    def prune_name_from_other_authors_list(fac_name, authors_list, dblp_df):
        mod_fac_name = fac_name.replace("-", " ")
        mod_fac_name_list = mod_fac_name.split(" ")
        #print(mod_fac_name_list)
        best_i = 0
        best_count = 0

        for i in range(len(authors_list)):
            #check matchability
            author = authors_list[i]
            count = 0
            for w in mod_fac_name_list:
                if w in author:
                    count += 1
            if count > best_count:
                best_count = count
                best_i = i
        authors_list.remove(authors_list[best_i])
        return authors_list
    dblp_df.apply(lambda row: prune_name_from_other_authors_list(row["Faculty"], row["Other Authors"], dblp_df), axis = 1)
    print('Step 5 Completed\n')

    # Step 5.5: Output
    # Store dblp_df with pickle
    dblp_12k_processed_df = dblp_df
    dblp_df.to_csv(r'dblp_df.csv', index = False)
    with open('dblp_12k_processed_df.pkl', 'wb') as f:
        pickle.dump(dblp_df, f)
    # print('Step 5.5 Completed: Output Complete\n')

    # Step 6: Retrieve unique, non-SCSE authors with distinguishable names
    def filter_unique_authors(dblp_df):
        # Iterate to fill up list of other authors
        other_authors_list = []
        for i, row in dblp_df.iterrows():
            for each in row['Other Authors']:
                other_authors_list.append(each)

        # Convert to retrieve only unique author entries
        other_authors_set = set(other_authors_list)
        unique_authors_list = list(other_authors_set)

        # Iterate to retrieve unique authors with distinguishable names
        unique_authors_numberless_list = []
        for each in unique_authors_list:
            if re.search(r'\d', each) == None:
                unique_authors_numberless_list.append(each)
        return unique_authors_numberless_list
    unique_authors_numberless_list = filter_unique_authors(dblp_df)
    def remove_scse(main_list):

        faculty_list = dblp_df['Faculty'].tolist()
        unique_faculty_list = list(set(faculty_list))

        for each in unique_faculty_list:

            mod_fac_name = each.replace("-", " ")
            mod_fac_name_list = mod_fac_name.split(" ")
            #print(mod_fac_name_list)
            best_i = 0
            best_count = 0

            for i in range(len(main_list)):
                #check matchability
                author = main_list[i]
                count = 0
                for w in mod_fac_name_list:
                    if w in author:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_i = i
            main_list.remove(main_list[best_i])
        return main_list
    unique_authors_numberless_scseless_list = remove_scse(unique_authors_numberless_list)
    print('Step 6 Completed\n')

    # Step 7: Retrieve faculty members of interest
    def retrieve_thousand_faculty(dblp_df):
        ssb_df = dblp_df[dblp_df['Faculty'] == 'Sourav S Bhowmick']

        # Iterate to fill up list of other authors
        ssb_other_authors_list = []
        for i, row in ssb_df.iterrows():
            for each in row['Other Authors']:
                ssb_other_authors_list.append(each)

        # Convert to retrieve only unique author entries
        ssb_unique_authors_list = list(set(ssb_other_authors_list))

        # Iterate to retrieve unique authors with distinguishable names
        ssb_unique_authors_numberless_list = []
        for each in ssb_unique_authors_list:
            if re.search(r'\d', each) == None:
                ssb_unique_authors_numberless_list.append(each)

        # Cross-compare and retrieve only non-scse sourav pals
        sourav_pal_list = []
        for each in unique_authors_numberless_scseless_list:
            for ssb_each in ssb_unique_authors_numberless_list:
                if ssb_each == each:
                    sourav_pal_list.append(ssb_each)

        # Retrieve non-sourav pal authors from main list
        remainder_authors_list = list(set(unique_authors_numberless_scseless_list).symmetric_difference(set(sourav_pal_list)))

        # randomly chosen remainder people
        random_chosen_list = random.sample(remainder_authors_list, 1000-len(sourav_pal_list))

        return sourav_pal_list + random_chosen_list
    thousand_apostles_list = retrieve_thousand_faculty(dblp_df)
    #print(len(thousand_apostles_list))
    print('Step 7 Completed\n')

    # Step -: Store thousand faculty members with pickle
    with open('1000_faculty.pkl', 'wb') as f:
        pickle.dump(thousand_apostles_list, f)
        print('Step 8 Completed\n')

    # Step 9: From here on out, we need to write code do perform 2 kinds of actions:
    # - load faculty graph and create initial graph of 85 people
    # - load faculty graph and create augmented graph of 1085 people
    # Step 9: Load faculty members

    # Step 8.1: Create Faculty Graph
    def create_faculty_network():
        # Import faculty details into df
        faculty_df = pd.read_excel('Faculty.xlsx')

        # Select relevant columns
        faculty_df = faculty_df[['Faculty', 'Position', 'Gender', 'Management', 'DBLP', 'Area']]

        # Create dictionary
        faculty_dict = faculty_df.to_dict('index')

        # Declare empty List
        faculty_list = []

        # Iterate over faculty_dict to fill up faculty_list
        for each in faculty_dict.items():

            node_no = each[0]
            faculty = each[1]['Faculty']
            position = each[1]['Position']
            gender = each[1]['Gender']
            management = each[1]['Management']
            dblp = each[1]['DBLP']
            area = each[1]['Area']

            faculty_list.append((node_no, {'Faculty': faculty}))
            faculty_list.append((node_no, {'Position': position}))
            faculty_list.append((node_no, {'Gender': gender}))
            faculty_list.append((node_no, {'Management': management}))
            faculty_list.append((node_no, {'DBLP': dblp}))
            faculty_list.append((node_no, {'Area': area}))

        # Declare empty new graph for faculty network
        faculty_graph = nx.MultiGraph()

        # Fill up empty graph w/ faculty_list
        faculty_graph.add_nodes_from(faculty_list)

        return faculty_graph
    faculty_graph = create_faculty_network()
    print('Step 9 Completed\n')

    # Step 8.2: Plot and Save Faculty Network
    def save_plot_faculty_network(faculty_graph):

        # Set figure for graph
        plt.figure(figsize=(35, 20))

        # Draw the graph
        nx.draw(faculty_graph, with_labels=True, font_size=10,
                node_color='red', font_color='white', edge_color='grey', node_size=250)

        # Save the graph
        plt.savefig("collab_graph.png", dpi=326)
    print('Step 10 Completed\n')

    # Step KW
    def kw_analysis(dblp_df):
        dblp_multi_df = dblp_df[dblp_df['key'].duplicated(keep=False)]
        dblp_multi_df.sort_values(by=['key'])

        # get categorical uniques in dataframe
        categorical_list = dblp_multi_df['key'].drop_duplicates().tolist()

        # create list to store key:f_index_list mappins
        key_findex_list = []

        # use categorical uniques to return df records
        for each in categorical_list:
            # extract f_index values from returned df records into a list
            mappings = dblp_multi_df[dblp_multi_df['key'] == each]['f_index'].tolist()
            # create unique pair-wise combinations for mappings (needed for networkx)
            mappings_pair = list(combinations(mappings, 2))
            year = dblp_multi_df[dblp_multi_df['key'] == each]['Year'].iloc[0]
            key_findex_list.append([mappings_pair, each, year])

        # list comprehensions
        pairings = np.array(key_findex_list)[:,[0,2]]
        distinct_paired_edges = [x for x in pairings if x[0][0][0] != x[0][0][1]]
        distinct_paired_edges = [[list(set(x[0])), x[1]] for x in distinct_paired_edges]
        distinct_paired_edges = [[[y, x[1]]for y in x[0]] for x in distinct_paired_edges]
        # removing the flipped ones i.e I only want (0,16) and not (16,0)
        distinct_paired_edges = [[y for y in x if y[0][0] < y[0][1]] for x in distinct_paired_edges]
        distinct_paired_edges = [[x[0][0], x[0][0], x[0][1]] for x in distinct_paired_edges]
        edge_year_pairings = [[x[0][0], x[0][1], x[1]] for x in distinct_paired_edges]
        edge_year_pairings = [tuple(x) for x in edge_year_pairings]

        # save key_findex_list into pickle file for easy replication
        file_name = "edge_year_pairings.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(edge_year_pairings, open_file)
        open_file.close()

        # iterate over key_findex_list to populate initial network
        for edge in edge_year_pairings:
            faculty_graph.add_edge(edge[0], edge[1], label = edge[2])

        # save faculty_graph into pickle file for easy replication
        file_name = "faculty_graph.pkl"
        open_file = open(file_name, "wb")
        pickle.dump(faculty_graph, open_file)
        open_file.close()

    # Step 9.1: Create Augmented Faculty Graph
    def create_augmented_faculty_network(dblp_12k_processed_df, thousand_apostles_list):

        # Import faculty details into df
        faculty_df = pd.read_excel('Faculty.xlsx')

        # Select relevant columns
        faculty_df = faculty_df[['Faculty', 'Position', 'Gender', 'Management', 'DBLP', 'Area']]

        # Augment Previously created faculty_df w/ 1000 apostles
        for new_faculty in thousand_apostles_list:
            row = pd.Series([new_faculty, '-', '-', '-', '-', '-'], index=faculty_df.columns)
            faculty_df = faculty_df.append(row, ignore_index=True)

        # Create dictionary
        faculty_dict = faculty_df.to_dict('index')

        # Declare empty List
        faculty_list = []

        # Iterate over faculty_dict to fill up faculty_list
        for each in faculty_dict.items():

            node_no = each[0]
            faculty = each[1]['Faculty']
            position = each[1]['Position']
            gender = each[1]['Gender']
            management = each[1]['Management']
            dblp = each[1]['DBLP']
            area = each[1]['Area']

            faculty_list.append((node_no, {'Faculty': faculty}))
            faculty_list.append((node_no, {'Position': position}))
            faculty_list.append((node_no, {'Gender': gender}))
            faculty_list.append((node_no, {'Management': management}))
            faculty_list.append((node_no, {'DBLP': dblp}))
            faculty_list.append((node_no, {'Area': area}))

        # Declare empty new graph for faculty network
        faculty_graph_1k = nx.MultiGraph()

        # Fill up empty graph w/ faculty_list
        faculty_graph_1k.add_nodes_from(faculty_list)

        # Assign f_index to all augmented faculty members list
        faculty_df.insert(0, 'f_index', faculty_df.index.values.tolist())

        # Segment to use previously-extracted DBLP raw data to map collaborations between faculty staff and 1k apostles

        # Import dblp_df_2.csv as DF
        dblp_df = dblp_12k_processed_df

        unique_oa_list = dblp_df['Other Authors'].tolist()
    faculty_1k_graph = create_augmented_faculty_network
    print('Step 11 Completed\n')

    # Step 9.2: Plot and Save Faculty Network
    def save_plot_faculty_1k_network(faculty_1k_graph):

        plt = faculty_1k_graph

        # Set figure for graph
        plt.figure(figsize=(200, 100))

        # Draw the graph (with isolates removed)
        nx.spring_layout(faculty_graph_1k, k=0.25, iterations=20)
        nx.draw(faculty_graph_1k, with_labels=True, font_size=20,
                node_color='red', font_color='white', edge_color='grey', node_size=1500)

        # Save the graph
        plt.savefig("faculty_graph_1k.png", dpi=50)
    print('Step 12 Completed\n')

    # Step 13: Output df and graph files
    faculty_df = pd.read_excel('Faculty.xlsx')
    faculty_df.to_csv('faculty_df.csv', index=False)

    try:
        with open('faculty_1k_graph.pkl', 'wb') as f:
            pickle.dump(faculty_1k_graph, f)
    except:
        print()
    print('Step 13 Completed\n')
