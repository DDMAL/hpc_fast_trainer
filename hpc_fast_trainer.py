from rodan.jobs.base import RodanTask
from time import sleep
from uuid import uuid4
import base64
import json
import os
import pika


class HPCFastTrainer(RodanTask):
    name = "Training model for Patchwise Analysis of Music Document - HPC"
    author = "Juliette Regimbal"
    description = "Performs the fast trainer job on Compute Canada Cedar"
    enabled = True
    category = "OMR - Layout analysis"
    interactive = False

    settings = {
        'title': 'Training parameters',
        'type': 'object',
        'job_queue': 'Python3',
        'properties': {
            'Maximum number of training epochs': {
                'type': 'integer',
                'minimum': 1,
                'default': 10
            },
            'Patch height': {
                'type': 'integer',
                'minimum': 64,
                'default': 256
            },
            'Patch width': {
                'type': 'integer',
                'minimum': 64,
                'default': 256
            },
            'Maximum time (D-HH:MM)': {
                'type': 'string',
                'default': '0-03:00'
            },
            'Maximum memory (MB)': {
                'type': 'integer',
                'minimum': 1024,
                'default': 3072
            },
            'CPUs': {
                'type': 'integer',
                'minimum': 1,
                'default': 2
            }
        }
    }

    input_port_types = (
        {'name': 'Image', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgb+png', 'image/rgb+jpg']},
        {'name': 'rgba PNG - Background layer', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Music symbol layer', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Staff lines layer', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Text', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgba+png']},
        {'name': 'rgba PNG - Selected regions', 'minimum': 1, 'maximum': 1, 'resource_types': ['image/rgba+png']}
    )

    output_port_types = (
        {'name': 'Background Model', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Music Symbol Model', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Staff Lines Model', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
        {'name': 'Text Model', 'minimum': 1, 'maximum': 1, 'resource_types': ['keras/model+hdf5']},
    )

    def run_my_task(self, inputs, settings, outputs):
        input = {}
        with open(inputs['Image'][0]['resource_path'], 'rb') as f:
            input['Image'] = base64.encodebytes(f.read()).decode('utf-8')
        with open(inputs['rgba PNG - Background layer'][0]['resource_path'], 'rb') as f:
            input['Background'] = base64.encodebytes(f.read()).decode('utf-8')
        with open(inputs['rgba PNG - Music symbol layer'][0]['resource_path'], 'rb') as f:
            input['Music Layer'] = base64.encodebytes(f.read()).decode('utf-8')
        with open(inputs['rgba PNG - Staff lines layer'][0]['resource_path'], 'rb') as f:
            input['Staff Layer'] = base64.encodebytes(f.read()).decode('utf-8')
        with open(inputs['rgba PNG - Text'][0]['resource_path'], 'rb') as f:
            input['Text'] = base64.encodebytes(f.read()).decode('utf-8')
        with open(inputs['rgba PNG - Selected regions'][0]['resource_path'], 'rb') as f:
            input['Selected Regions'] = base64.encodebytes(f.read()).decode('utf-8')

        message_dict = {
            'inputs': input,
            'settings': settings
        }
        message = json.dumps(message_dict)

        credentials = pika.PlainCredentials(os.environ['HPC_RABBITMQ_USER'], os.environ['HPC_RABBITMQ_PASSWORD'])
        parameters = pika.ConnectionParameters(os.environ['HPC_RABBITMQ_HOST'], 5672, '/', credentials)
        result_dict = None
        with pika.BlockingConnection(parameters) as conn:
            # Open Channel
            channel = conn.channel()
            channel.queue_declare(queue='hpc-jobs')
            channel.queue_declare(queue='hpc-results')
            # Declare anonymous reply queue
            #result = channel.queue_declare(queue='', exclusive=True)
            #callback_queue = result.method.queue
            callback_queue = 'hpc-results'
            correlation_id = str(uuid4())
            # Send Message
            channel.basic_publish(
                exchange='',
                routing_key='hpc-jobs',
                properties=pika.BasicProperties(
                    reply_to=callback_queue,
                    correlation_id=correlation_id
                    ),
                body=message
            )

            # Check for response
            message_received = False
            body = None
            while not message_received:
                # Get message from queue
                method_frame, header_frame, body = channel.basic_get(callback_queue)
                if method_frame and correlation_id == header_frame.correlation_id:
                    message_received = True
                    channel.basic_ack(method_frame.delivery_tag)
                    result_dict = json.loads(body.decode('utf-8'))
                else:
                    sleep(60)

        with open(outputs['Background Model'][0]['resource_path'], 'wb') as f:
            f.write(base64.decodebytes(result_dict['Background Model'].encode('utf-8')))
        with open(outputs['Music Symbol Model'][0]['resource_path'], 'wb') as f:
            f.write(base64.decodebytes(result_dict['Music Symbol Model'].encode('utf-8')))
        with open(outputs['Staff Lines Model'][0]['resource_path'], 'wb') as f:
            f.write(base64.decodebytes(result_dict['Staff Lines Model'].encode('utf-8')))
        with open(outputs['Text Model'][0]['resource_path'], 'wb') as f:
            f.write(base64.decodebytes(result_dict['Text Model'].encode('utf-8')))

        return True
