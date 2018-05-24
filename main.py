# Set up REST server with one endpoint for sentence autocomplete

import tornado.ioloop
import tornado.web
from autocomplete import autocomplete


class autocomplete_handler(tornado.web.RequestHandler):
    """
    TODO: add helpful comments
    """

    def get(self):
        """
        TODO: add helpful comments
        """
        args = self.get_arguments("q")
        # using args just to test that server works. TODO: make an autocomplete function to use here
        response = {"Completions": autocomplete(args)}
        self.write(response)
        self.write("\n")


def make_app():
    """
    TODO: add helpful comments
    """
    return tornado.web.Application([
        (r"/autocomplete", autocomplete_handler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(13000)
    tornado.ioloop.IOLoop.current().start()
