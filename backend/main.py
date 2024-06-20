from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from services.genai import YoutubeProcessor, GeminiProcessor


class VideoAnalysisRequest(BaseModel):
    youtube_link: HttpUrl
    # advanced setting

app = FastAPI()

#configure cor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers =["*"],
)

@app.post("/analyze_video")
def analyze_video(request: VideoAnalysisRequest):
    # doing  the analysis
    
    result = YoutubeProcessor().retrieve_youtube_documents(str(request.youtube_link), verbose=True)
    genai_processor = GeminiProcessor(
        model_name="gemini-pro",
        project= "education-mission",
        location = "europe-west2"
    )

    summary = genai_processor.generate_document_summary(result, verbose =True)

    '''
    author = result[0].metadata["author"]
    length = result[0].metadata["length"]
    title = result[0].metadata["title"]
    total_size = len(result)
    

    return {
        #"youtube_link": request.youtube_link
        "author": author,
        "length": length,
        "title": title,
        "total_size": total_size
    }

    '''
    #return result

    return {
        "summary" :summary
    }


@app.get("/")
def health():
    return {"status": "ok"}



'''
    from langchain_community.document_loaders import YoutubeLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    loader = YoutubeLoader.from_youtube_url(str(request.youtube_link), add_video_info = True)
    docs = loader.load()
    print(f"On Load: {type(docs)}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)#chunk_size=1000, chunk_overlap=0
    result  = text_splitter.split_documents(docs)

    print(f"{type(result)}")
'''