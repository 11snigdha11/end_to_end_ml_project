#building our application as a package itself
# from setuptools import find_packages,setup
# from typing import List

# HYPEN_E_DOT='-e .'
# def get_requirements(file_path:str)->List[str]:


#     requirements=[]
#     with open(file_path) as file_obj:
#         requirements=file_obj.readlines()
#         requirements=[req.replace("\n","") for req in requirements]
#         #requirements=[req.strip() for req in requirements]

#         #inorder to not read last line
#         if HYPEN_E_DOT in requirements:

#             requirements.remove(HYPEN_E_DOT)
#     return requirements

# setup(
#     name='mymlproject',
#     version='0.0.1',
#     author='snigdha',
#     author_email='snigdha.p2004@gmail.com',
#     packages=find_packages(),
#     #install_requires=['pandas','numpy','seaborn']
#     install_requires=get_requirements('requirements.txt')
from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        
        cleaned_requirements = []
        for req in requirements:
            # Strip spaces/newlines, and ignore anything after a '#' comment
            clean_req = req.split('#')[0].strip()
            
            # Only add to list if it's not an empty string
            if clean_req:
                cleaned_requirements.append(clean_req)

        if HYPEN_E_DOT in cleaned_requirements:
            cleaned_requirements.remove(HYPEN_E_DOT)
            
    return cleaned_requirements

setup(
    name='mymlproject',
    version='0.0.1',
    author='snigdha',
    author_email='snigdha.p2004@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)










