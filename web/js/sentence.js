// Pure text-composition logic. No DOM access, so it can be unit tested in Node.

const SENTENCE_END = /[.?!]$/;

export class Composer {
  constructor() {
    this.text = "";
  }

  #shouldCapitalize() {
    const t = this.text.replace(/ +$/, "");
    return t === "" || SENTENCE_END.test(t) || t.endsWith("\n");
  }

  addLetter(letter) {
    const ch = letter.toLowerCase();
    this.text += this.#shouldCapitalize() ? ch.toUpperCase() : ch;
  }

  addSpace() {
    if (this.text === "" || /[ \n]$/.test(this.text)) return;
    this.text += " ";
  }

  addPunctuation(mark) {
    if (this.text === "") return;
    this.text = this.text.replace(/ +$/, "");
    if (/[.?!,]$/.test(this.text)) return;
    this.text += mark;
  }

  addNewline() {
    if (this.text === "" || this.text.endsWith("\n")) return;
    this.text = this.text.replace(/ +$/, "") + "\n";
  }

  backspace() {
    this.text = this.text.slice(0, -1);
  }

  clear() {
    this.text = "";
  }

  // The sentence currently being written (text after the last . ? ! or newline).
  currentSentence() {
    const parts = this.text.split(/(?<=[.?!])\s+|\n/);
    return (parts[parts.length - 1] ?? "").trim();
  }

  endsSentence() {
    return SENTENCE_END.test(this.text.trimEnd());
  }
}
