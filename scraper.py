import requests
from bs4 import BeautifulSoup

URL = "https://www.indiatoday.in/good-news?page=5"
page = requests.get(URL)

soup = BeautifulSoup(page.content, "html.parser")
job_elements = soup.find_all("div", class_="catagory-listing")
for job_element in job_elements:
  headline = job_element.find("h2", class_="")
  print(headline.text + ".")
