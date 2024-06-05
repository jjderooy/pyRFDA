import numpy as np

class Geometry:

    def shape(self):
        return self._shape

    def rect(self, L, t, b, m):
        '''
        Inputs:
          L:  [m]  Length
          t:  [m]  Thickness
          b:  [m]  Width
          m:  [kg] Mass

        Desc:
        Define a rectangular geometry for RFDA. Error checks geometry
        dimensions to ensure it meets criteria for RFDA.
        '''

        self._shape = 'rect'
        self._L = L 
        self._t = t 
        self._b = b 
        self._m = m 

        if L/t < 20:
            raise ValueError("Warning: For rect geometry L/t must be greater than 20")

    def rod(self, L, t, d, m):
        '''
        Inputs:
          L:  [m]  Length
          t:  [m]  Thickness
          d:  [m]  Diameter
          m:  [kg] Mass

        Desc:
          Define a cylindrical geometry for RFDA. Error checks geometry
          dimensions to ensure it meets criteria for RFDA.
        '''

        self._shape = 'rod'
        self._L = L # [m]  Length
        self._t = t # [m]  Thickness
        self._d = d # [m]  Diameter
        self._m = m # [kg] Mass

        if L/d < 20:
            raise ValueError("Warning: For rod geometry L/d must be greater than 20")
            pass

    def disc(self, t, d, m):
        '''
        Inputs:
          t:  [m]  Thickness
          d:  [m]  Diameter
          m:  [kg] Mass

        Desc:
          Define a disc (like a coin) geometry for RFDA. Error checks
          geometry dimensions to ensure it meets criteria for RFDA.
        '''

        raise ValueError("Not implemented! Add error check on disk!")
  
        self._shape = 'disc'
        self._t = t # [m]  Thickness
        self._d = d # [m]  Diameter
        self._m = m # [kg] Mass

        # TODO add a check for geometry and throw a ValueError if needed

    def flexural_freq(self, E):
        '''
        Inputs:
          E:   [Pa] Elastic modulus of material.
  
        Outputs:
          f_f: [Hz] Resonant frequency of the flexural mode of vibration
  
        Desc:
          The flexural resonant frequency f_f of a sample can be computed from
          the elastic modulus E of the  material and its geometry. More info:
          https://en.wikipedia.org/wiki/Impulse_excitation_technique
        '''
      
        match self._shape:
            case 'rect':
                # Correction factor
                T = 1 + 6.585*np.power((self._t/self._L), 2)
      
                f_f = np.sqrt(E/(T*0.9465)* \
                    (self._b/self._m)*np.power(self._t/self._L,3))
                return f_f
  
            case 'rod':
                #Correction factor
                T = 1+4.939*np.power(self._d/self._L,2)

                f_f = np.sqrt(E/(1.6067* \
                                 (np.power(self._L,3)/np.power(self._d,4)) \
                                 *self._m*T))
                return f_f

            case 'disc':
                raise NotImplementedError("Not implemented for disk!")

            case _:
                raise ValueError("_shape must be one of 'rect', 'rod', 'disc'")

    def torsional_freq(self, G):
        '''
        Inputs:
          G:   [Pa] Shear modulus of material.
  
        Outputs:
          f_f: [Hz] Resonant frequency of the torsional mode of vibration
  
        Desc:
          The torsional resonant frequency f_t of a sample can be computed from
          the shear modulus G of the  material and its geometry. More info:
          https://en.wikipedia.org/wiki/Impulse_excitation_technique
        '''
  
        match self._shape:
            case 'rect':
                # Correction factor
                R = ((1+np.power(self._b/self._t, 2)) / \
                    (4-2.521*(self._t/self._b) * \
                    (1-(1.991/(np.exp(np.pi*self._b/self._t) + 1))))) * \
                    (1+0.00851*np.power(self._b/self._L, 2)) - \
                    (0.060*np.power(self._b/self._L, 3/2)*np.power(self._b/self._t-1,2))
      
                f_t = np.sqrt((G/R)*(self._b*self._t/(4*self._L*self._m)))
                return f_t
  
            case 'rod':
                # No correction factor needed
                f_t = np.sqrt(G/(16*self._L*self._m/(np.pi*np.power(self._d,2))))
                return f_t

            case 'disc':
                raise NotImplementedError("Not implemented for disk!")

            case _:
                raise ValueError("_shape must be one of 'rect', 'rod', 'disc'")

    def node_spacing(self):
        '''
        Outputs:
          node_spacing: (first_node, node_spacing) [m] 
                        Tuple with the first value being the location of the
                        node from the end of the sample, and the second being
                        the space from that node to the next.
  
        Desc:
            Samples that are resonating should be held at their vibrational
            nodes. This varies depending on the geometry.
        '''
  
        match self._shape:
            case 'rect':
                # Each node is just 0.244L from the end of the bar
                # http://hyperphysics.phy-astr.gsu.edu/hbase/Music/barres.html
                first_node   = 0.244*self._L
                node_spacing = (1-2*0.244)*self._L
                return (first_node, node_spacing)
  
            case 'rod':
                # Each node is just 0.244L from the end of the bar
                # http://hyperphysics.phy-astr.gsu.edu/hbase/Music/barres.html
                first_node   = 0.244*self._L
                node_spacing = (1-2*0.244)*self._L
                return (first_node, node_spacing)

            case 'disc':
                raise NotImplementedError("Not implemented for disk!")

            case _:
                raise ValueError("_shape must be one of 'rect', 'rod', 'disc'")
