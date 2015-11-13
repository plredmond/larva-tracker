#!/usr/bin/env python2
from __future__ import \
    ( nested_scopes
    , generators
    , division
    , absolute_import
    , with_statement
    , print_function
    , unicode_literals
    )

# https://eleanormaclure.files.wordpress.com/2011/03/colour-coding.pdf
# https://en.wikipedia.org/wiki/Help:Distinguishable_colors

__alphabet = \
    [ ('(white background assumed)', (255,255,255))
    , ('Amethyst', (240,163,255))
    , ('Blue', (0,117,220))
    , ('Caramel', (153,63,0))
    , ('Damson', (76,0,92))
    , ('Ebony', (25,25,25))
    , ('Forest', (0,92,49))
    , ('Green', (43,206,72))
    , ('Honeydew', (255,204,153))
    , ('Iron', (128,128,128))
    , ('Jade', (148,255,181))
    , ('Khaki', (143,124,0))
    , ('Lime', (157,204,0))
    , ('Mallow', (194,0,136))
    , ('Navy', (0,51,128))
    , ('Orpiment', (255,164,5))
    , ('Pink', (255,168,187))
    , ('Quagmire', (66,102,0))
    , ('Red', (255,0,16))
    , ('Sky', (94,241,242))
    , ('Turquoise', (0,153,143))
    , ('Uranium', (224,255,102))
    , ('Violet', (116,10,255))
    , ('Wine', (153,0,0))
    , ('Xanthin', (255,255,128))
    , ('Yellow', (255,255,0))
    , ('Zinnia', (255,80,5))
    ]

alphabet = lambda: __alphabet[:]

class ResourceLibrary(object):
    '''Resource pool request resolution.

       Not threadsafe.
    '''
    class LibraryIsEmpty(Exception): pass
    class ResourceInUseException(Exception): pass

    def __init__(self, stock):
        '''{ResourceName: Resource} -> ResourceLibrary'''
        self._stocked = dict(stock) # {ResourceName: Resource}
        self._borrows = dict() # {hashable: (ResourceName, Resource)}

    def __reserve_resource(self, resource_name):
        try:
            return self._stocked.pop(resource_name)
        except KeyError:
            raise self.ResourceInUseException(resource_name)

    def __reserve_any_resource(self):
        try:
            return self._stocked.popitem()
        except KeyError:
            raise self.LibraryIsEmpty()

    def __check_out_resource(self, borrower, resource_name, resource):
        self._borrows[borrower] = (resource_name, resource)

    def kariru(self, borrower, resource_name=None):
        '''hashable, Maybe<ResourceName> -> Resource '''
        # give 'em the resource in case they forgot
        item = self.query(borrower)
        if item is not None:
            return item[1]
        # give 'em a new item if possible
        if resource_name is None:
            resource_name, resource = self.__reserve_any_resource()
        else:
            resource = self.__reserve_resource(resource_name)
        self.__check_out_resource(borrower, resource_name, resource)
        return resource

    def query(self, borrower):
        '''hashable -> Maybe<(ResourceName, Resource)>'''
        return self._borrows.get(borrower)

    def kaesu(self, borrower):
        resource_name, resource = self._borrows.pop(borrower)
        self._stocked[resource_name] = resource

# eof
