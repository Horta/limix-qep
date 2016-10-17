# def _joint_update(self):
#     K = self.K()
#     m = self.m()
#     A2 = self._A2()
#     QB1Qt = self._Q0B1Q0t()
#
#     jtau = self._joint_tau
#     jeta = self._joint_eta
#
#     diagK = K.diagonal()
#     QB1QtA1 = ddot(QB1Qt, self._A1(), left=False)
#     jtau[:] = 1 / (A2 * diagK - A2 * dotd(QB1QtA1, K))
#
#     Kteta = K.dot(self._sitelik_eta)
#     jeta[:] = A2 * (m - QB1QtA1.dot(m) + Kteta - QB1QtA1.dot(Kteta))
#     jeta *= jtau
