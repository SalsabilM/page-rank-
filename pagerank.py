import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85 #damping factor
SAMPLES = 10000 #number of samples to be used in the sampling method of PageRank.


def main():
    if len(sys.argv) != 2:
        sys.exit ("Usage:python pagerank.py corpus ")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict() #store the links found in each HTML file.

    # Extract all links from HTML files
    for filename in os.listdir(directory): #Lists all files in the specified directory
        if not filename.endswith(".html"): # Checks if the file has an .html extension.
            continue
        with open(os.path.join(directory, filename)) as f: # Opens the HTML file and reading. 
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]?)href=\"([^\"])\"", contents) # find all links in the HTML content.
            pages[filename] = set(links) - {filename} #Converts the list of links to a set to remove duplicates
                                                    #Stores the set of links in the pages dictionary with the filename as the key

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability damping_factor, choose a link at random
    linked to by page. With probability 1 - damping_factor, choose
    a link at random chosen from all pages in the corpus.
    """
    avail = corpus[page] #set of pages that the current page links to, as defined in the corpus.
    probability = {} # store the transition probabilities for each page in the corpus.
    if len(avail) == 0: #Checks if the current page has no outgoing links.
        prob_page = 1/len(corpus.keys())
    else:
        prob_page = (1-damping_factor)/len(corpus.keys())
        prob_link = damping_factor/len(avail)
        
    for page in corpus.keys():
        if page in avail:
            probability[page] = prob_page+prob_link
        else:
            probability[page] = prob_page
                
                
                
    return probability   
            
    #raise NotImplementedError

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling n pages
    according to transition model, starting with a page at random.
    
    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page = random.choice(list(corpus.keys()))#Selects a random page from the corpus to start the sampling process.
    sample = []#list to store the sequence of sampled pages.
    sample.append(page)#Adds the initial random page to the sample list.
    
    
    for i in range(n):
        current_page = sample[i]# Sets the current page
        probabilities = transition_model(corpus, current_page, damping_factor)
        next_page = np.random.choice(list(probabilities.keys()), p=list(probabilities.values())) #Selects the next page based on the computed probabilities
        sample.append(next_page)
    
    dict_page = {}# dictionary to store the estimated PageRank values for each page.
    for page in corpus.keys():
        dict_page[page] = sample.count(page)/len(sample) #Calculates the PageRank for each page as the proportion of times the page appears in the sample list.
    return dict_page
        
    
    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    
    pagerank = {} # store the PageRank value of each page.
    for page in corpus.keys():
        pagerank[page] = 1/len(corpus.keys())#Initializes the PageRank value of each page
    
    prev_rank = pagerank.copy()# copy of the initial PageRank values to keep track of the previous iteration's values.
    flg = dict([(key, False) for key in prev_rank.keys()])#flag whether the PageRank value of each page has converged
    
    while True:
        
        for page in corpus.keys():
            prob_1 = (1-damping_factor)/len(corpus.keys())
            
            prob_2 = 0 # Initialize the second part of the PageRank formula.
            for prev_page, page_links in corpus.items():
                if len(page_links) == 0: #If a page has no links, links to all pages and including itself.
                    page_links = corpus.keys()
                if page in page_links:
                    prob_2 += prev_rank[prev_page]/len(page_links)
                    
            final_prob = prob_1 + damping_factor*prob_2
            pagerank[page] = final_prob
              
        for page in prev_rank.keys():
            if abs(prev_rank[page] - pagerank[page]) < 0.001:
                flg[page] = True #Mark the page as converged 
            else:
                flg[page] = False #mark the page as not converged.
        if all(flg.values()):# Exit the loop if all pages have converged.
            break
        prev_rank = pagerank.copy()
    return pagerank
    raise NotImplementedError

if __name__ == "__main__":
    main()
