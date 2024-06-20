# Project-Untitled-AiB2024

## Project Description

- โมเดลแนะนำว่าควรปลูกพืชอะไร โดยเน้นพืชเศรษฐกิจของไทย
- สร้าง dataset จาก public data ของได้ ได้แก่ ข้อมูล soil series data จาก กรมพัฒนาที่ดิน ข้อมูลรายละเอียดของดิน จาก สำนักสำรวจดินและวิจัยทรัพยากรดิน และข้อมูลจากระบบการติดตามสถานการณ์เพาะปลูกพืชเศรษฐกิจในประเทศไทยจากข้อมูลดาวเทียม จาก GISTDA และนำมาสร้าง tabular data ด้วยการทำ grid และแบ่ง train valid test โดยใช้ spatial train valid test split
- วัดผลโดยใช้ top-k accuracy matric
XGBoost, Catboost, และ KNN (with standard scaler) ทำออกมาได้ดีพอ ๆ กัน โดยได้ top-k accuracy score (k=1) อยู่ที่ 80.6%, 83.2%, และ 88.9%
- เมื่อเทียบกับ Most Frequent Class ตามผลเฉลยที่เป็นไปได้มากที่สุด top-k accuracy score (k=1) อยู่ที่ 58.7%

### Medium 

https://medium.com/@tangnatta/%E0%B9%80%E0%B8%A3%E0%B8%B2%E0%B8%84%E0%B8%A7%E0%B8%A3%E0%B8%9B%E0%B8%A5%E0%B8%B9%E0%B8%81%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3%E0%B8%94%E0%B8%B5-w-summary-fa9d7661e138

## Code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tangnatta/Project-Untitled-AiB2024/blob/main/FinalPublicNotebook.ipynb) [Not well supported on Colab, but GitHub can't render this notebook so it better to preview on Colab] 

## Demo 

https://deploymentpy-nqg4jgxq37rjnpkmrpjqef.streamlit.app/
