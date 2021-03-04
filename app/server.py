import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from datetime import datetime

#export_file_url = 'http://zapp-brannigan.ge.issia.cnr.it/resnet-50-stage-2-2019-06-19_13.08.13-export.pkl'
#export_file_name = 'resnet-50-stage-2-2019-06-19_13.08.13-export.pkl'

#export_file_url = 'https://srv-file2.gofile.io/download/4azGQy/resnet-50-stage-2-2019-06-19_130813-export.pkl'
#export_file_name = 'resnet-50-stage-2-2019-06-19_130813-export.pkl'

#export_file_url = 'https://drive.google.com/uc?export=download&id=1zLjFtKOutBVkhK7IfUYgHuMWfJl8WQn4'
#export_file_name = 'uc?export=download&id=1zLjFtKOutBVkhK7IfUYgHuMWfJl8WQn4'

export_file_url = 'http://zapp-brannigan.ge.imati.cnr.it/chihuahua-or-muffin-resnet-models/resnet-50-stage-4-2019-07-02_17.51.51-pth-export-0.045576-error-rate.pkl'
export_file_name = 'resnet-50-stage-4.pkl'

classes = ['bagel', 'chihuahua', 'chocolate', 'dalmatian', 'dog', 'duckling', 'friedchicken', 'guacamole', 'icecream', 'icecreamcone', 'kitten', 'labradoodle', 'marshmallow', 'mop', 'muffin', 'painauchocolat', 'parrot', 'plantain', 'sharpei', 'sheepdog', 'shiba', 'sloth']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    fname = str(datetime.now()).replace(' ', '-')
    img.save('/tmp/saved-images/' + fname + '.jpg')
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=55561, log_level="info")
