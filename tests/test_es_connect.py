import requests
from requests.auth import HTTPBasicAuth

# 设置服务器的 URL
url = 'https://localhost:9200'

# 设置用户名和密码
username = 'elastic'
password = 'KHWwbn1uwYvp_4ADa4Rr'

# 指定 CA 证书的路径
ca_cert_path = '/Users/yuhao/Documents/elastic/http_ca.crt'

# 发送 GET 请求
try:
    response = requests.get(url, auth=HTTPBasicAuth(username, password), verify=ca_cert_path)
    response.raise_for_status()  # 检查响应是否成功
    print("Response Status:", response.status_code)
    print("Response Body:", response.text)
except requests.exceptions.HTTPError as errh:
    print("Http Error:", errh)
except requests.exceptions.ConnectionError as errc:
    print("Error Connecting:", errc)
except requests.exceptions.Timeout as errt:
    print("Timeout Error:", errt)
except requests.exceptions.RequestException as err:
    print("OOps: Something Else", err)
