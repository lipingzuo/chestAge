let tools = {
  // 打印功能，表标题、条数、数据、列标题
  print: (title, limit, list, header) => {
    var keys = []
    for (let key in list[0]) {
      keys.push(key)
    }
    var oPop = window.open('', 'oPop');
    var str = '<!DOCTYPE html>'
    str += '<html>'
    str += '<head>'
    str += '<meta charset="utf-8">'
    str += '<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">'
    str += '</head>'
    str += '<body>'
    //分页
    var page = list.length % limit ? parseInt(list.length / limit) + 1 : parseInt(list.length / limit)
    for (let k = 0; k < page; k++) {
      str += '<table style="width:100%;border-collapse :collapse;">'
      str += `<p style="text-align:center ">${title}</p>`
      //表头
      str += '<tr>'
      for (let i = 0; i < header.length; i++) {
        str += `<th style="height:40px;border :1px solid black;font-size:12px;text-align :center;">${header[i]}</th>`
      }
      str += '</tr>'

      //表体 每页的行数
      var hang = (k + 1) * limit <= list.length ? limit : list.length - limit * k
      for (let i = 0; i < hang; i++) {
        str += '<tr>'
        for (let j = 0; j < keys.length; j++) {
          str += `<td style="height:40px;border :1px solid black;font-size:12px;text-align :center;">${list[k * limit + i][keys[j]]}</td>`
        }
        str += '</tr>'
      }
      str += '</table>'
      str += '<div style="page-break-after:always"></div>'
    }

    str += '</body>'
    str += '</html>'
    oPop.document.write(str);
    oPop.print();
    oPop.close();
  }
}

module.exports = tools