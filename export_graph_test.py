import mock
import unittest
from unittest.mock import patch, MagicMock

import export_graph

class TestExportGraph(unittest.TestCase):

    @patch('export_graph.CycleGAN')
    def test_checkpoint_create(self, m_CycleGAN):
        m_CycleGAN.return_value = MagicMock()
        with mock.patch('export_graph.tf') as m_tf:
            export_graph.export_graph("apple2orange.pb",  XtoY=True)

        m_tf.train.write_graph.assert_called_with(
            m_tf.graph_util.convert_variables_to_constants(),
            'pretrained',
            'apple2orange.pb',
            as_text=False
        )

if __name__ == '__main__':
    unittest.main()
