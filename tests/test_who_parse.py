import requests
from bs4 import BeautifulSoup
import re

url = 'https://atcddd.fhi.no/atc_ddd_index/?name=oxycodone%20acetaminophen&showdescription=no'
print(f'Testing: {url}\n')

r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
tables = soup.find_all('table')

print(f'Found {len(tables)} tables\n')

for i, table in enumerate(tables):
    first_row = table.find('tr')
    if first_row:
        cells = first_row.find_all('td')
        print(f'Table {i}: {len(cells)} cells in first row')
        if len(cells) >= 2:
            print(f'  Cell 0: "{cells[0].get_text(strip=True)}"')
            print(f'  Cell 1: "{cells[1].get_text(strip=True)}"')
            
            # Check if looks like ATC code
            code = cells[0].get_text(strip=True)
            pattern = r'^[A-Z]\d{2}[A-Z]{2}\d{2}$'
            if re.match(pattern, code):
                print(f'  ✓ Valid ATC code!')
        print()
