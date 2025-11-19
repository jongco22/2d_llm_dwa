# LLM-DWA in 2D Simulation

## Execution
```bash
git clone https://github.com/jongco22/2d_llm_dwa.git
cd 2d_llm_dwa
```
Create a `.env` file in the root directory of the project and add your API key inside the file as follows:

```
API_KEY=your_api_key_here
```
Open the `api.py` file and modify the `dotenv_path` variable to match the location of your `.env` file:

```
dotenv_path = '.env file path'
```

### start
```
python <file_name>.py
```

## DWA
<img width="686" height="659" alt="fig4a" src="https://github.com/user-attachments/assets/282bf6f9-4194-48f4-8115-f0063bc4d550" />

## Dijkstra + DWA
<img width="688" height="659" alt="fig4b" src="https://github.com/user-attachments/assets/e4389d61-8f38-4d7d-9ab1-d45cff39f908" />

## LLM-DWA(Ours)
<img width="688" height="659" alt="fig4c" src="https://github.com/user-attachments/assets/11ee429c-cc5c-4fec-80a7-689a7f6962d0" />
