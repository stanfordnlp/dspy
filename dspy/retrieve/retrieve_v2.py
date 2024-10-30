import logging
import os

import cloudpickle

from dspy.utils.callback import with_callbacks

logger = logging.getLogger(__name__)


class RetrieveV2:
    def __init__(self, index=None, documents=None, build_index=False, callbacks=None):
        self.build_index = build_index
        self.documents = documents
        self.index = index
        if self.build_index:
            self.index = self.build_local_index(documents)

    def save(self, path):
        if self.build_index:
            with open(os.path.join(path, "documents.pkl"), "wb") as file:
                cloudpickle.dump(self.documents, file)

    def load(self, path):
        if self.build_index:
            file_path = os.path.join(path, "documents.pkl")
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist, nothing to load.")
                return

            with open(file_path, "rb") as file:
                self.documents = cloudpickle.load(file)
            self.index = self.build_local_index(self.documents)

    def build_local_index(self, documents):
        pass

    @with_callbacks
    def __call__(self, query, k=None, **kwargs):
        return self.forward(query, k=k, **kwargs)

    def forward(self, query, k=None, **kwargs):
        raise NotImplementedError
