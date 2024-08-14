from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By


# Path to the GeckoDriver executable
geckodriver_path = '/snap/bin/geckodriver'

# Set up Firefox options
firefox_options = Options()
firefox_options.add_argument('--headless')  # Run Firefox in headless mode

# Create a new instance of the Firefox driver
service = Service(geckodriver_path)
driver = webdriver.Firefox(service=service, options=firefox_options)

# URL of the website to visit
url = "https://datasetninja.com/crop-weed-detection#download"

# Navigate to the website
driver.get(url)



buttons = driver.find_elements(By.TAG_NAME, "button")

# iterate through all buttons to select which button to click on
print(f"Found {len(buttons)} buttons")
for button in buttons:
    if (button.text == ""):
        continue

    print(button.text)
    yn = input("click button [y/n]:\n")

    if (yn == 'y' or yn == 'Y'):
        button.click()
        break



# Close the browser
driver.quit()
